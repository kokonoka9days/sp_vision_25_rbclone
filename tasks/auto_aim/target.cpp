#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

constexpr double TOWTER_ARMOR_DH = 0.108; //前哨站两个装甲板之间的最短高低差m
constexpr double TOWER_ARMOR_DTB = 0.16;  //前哨装甲大跳变m
constexpr double TOWER_ARMOR_XTB = 0.05;  //前哨装甲小跳变m

namespace auto_aim
{

Eigen::VectorXd Target::mix_states(const Eigen::VectorXd& xa, const Eigen::VectorXd& xb, double wa, double wb) const {
  Eigen::VectorXd mixed = wa * xa + wb * xb;
  // 第7个状态是Yaw角(索引6)，直接按权重相加在跨越 π/-π 时会出错，需要通过差值计算
  double diff = tools::limit_rad(xa[6] - xb[6]);
  mixed[6] = tools::limit_rad(xb[6] + wa * diff);
  return mixed;
}

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  if(name == ArmorName::outpost){
    tower_armor_hs[0] = center_z;
  }

  cam_is_switch_time_point = std::chrono::steady_clock::time_point{};

  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  // 初始化 IMM 的两个 EKF
  ekf_1_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_2_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  combined_x_ = x0;

  // 初始化模型概率和状态转移矩阵
  mu_ << 0.5, 0.5;
  P_trans_ << 0.95, 0.05, 
              0.05, 0.95;
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_1_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ekf_2_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  combined_x_ = x0;

  mu_ << 0.5, 0.5;
  P_trans_ << 0.95, 0.05, 
              0.05, 0.95;
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };

  // --- IMM 步骤 1: 交互 (Interaction / Mixing) ---
  Eigen::Vector2d c;
  c[0] = P_trans_(0, 0) * mu_[0] + P_trans_(1, 0) * mu_[1];
  c[1] = P_trans_(0, 1) * mu_[0] + P_trans_(1, 1) * mu_[1];

  Eigen::Matrix2d mu_ij;
  mu_ij(0, 0) = P_trans_(0, 0) * mu_[0] / c[0];
  mu_ij(1, 0) = P_trans_(1, 0) * mu_[1] / c[0];
  mu_ij(0, 1) = P_trans_(0, 1) * mu_[0] / c[1];
  mu_ij(1, 1) = P_trans_(1, 1) * mu_[1] / c[1];

  Eigen::VectorXd x1 = ekf_1_.x;
  Eigen::VectorXd x2 = ekf_2_.x;
  Eigen::MatrixXd P1 = ekf_1_.P;
  Eigen::MatrixXd P2 = ekf_2_.P;

  Eigen::VectorXd x01 = mix_states(x1, x2, mu_ij(0, 0), mu_ij(1, 0));
  Eigen::VectorXd x02 = mix_states(x1, x2, mu_ij(0, 1), mu_ij(1, 1));

  auto normalize_diff = [](const Eigen::VectorXd& xa, const Eigen::VectorXd& xb) {
    Eigen::VectorXd dx = xa - xb;
    dx[6] = tools::limit_rad(dx[6]);
    return dx;
  };

  Eigen::VectorXd dx11 = normalize_diff(x1, x01);
  Eigen::VectorXd dx21 = normalize_diff(x2, x01);
  Eigen::VectorXd dx12 = normalize_diff(x1, x02);
  Eigen::VectorXd dx22 = normalize_diff(x2, x02);

  Eigen::MatrixXd P01 = mu_ij(0, 0) * (P1 + dx11 * dx11.transpose()) +
                        mu_ij(1, 0) * (P2 + dx21 * dx21.transpose());
  Eigen::MatrixXd P02 = mu_ij(0, 1) * (P1 + dx12 * dx12.transpose()) +
                        mu_ij(1, 1) * (P2 + dx22 * dx22.transpose());

  ekf_1_.x = x01; ekf_1_.P = P01;
  ekf_2_.x = x02; ekf_2_.P = P02;

  mu_ = c; // 预测阶段的模型概率更新

  // --- IMM 步骤 2: 生成不同模型的 Q 矩阵并预测 ---
  double v1_1, v2_1; // 模型 1: 平滑 (低噪声)
  double v1_2, v2_2; // 模型 2: 机动 (高噪声)
  
  if (name == ArmorName::outpost) {
    v1_1 = 5;     v2_1 = 0.05;
    v1_2 = 50;    v2_2 = 2.0; 
  } else {
    v1_1 = 50;    v2_1 = 50; 
    v1_2 = 1000;  v2_2 = 2000; // 应对急停急走和小陀螺的高倍率噪声
  }

  auto a_ = dt * dt * dt * dt / 4;
  auto b_ = dt * dt * dt / 2;
  auto c_ = dt * dt;

  auto build_Q = [&](double v1, double v2) -> Eigen::MatrixXd {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
    Q(0,0) = a_ * v1; Q(0,1) = b_ * v1; Q(1,0) = b_ * v1; Q(1,1) = c_ * v1; // X
    Q(2,2) = a_ * v1; Q(2,3) = b_ * v1; Q(3,2) = b_ * v1; Q(3,3) = c_ * v1; // Y
    Q(4,4) = a_ * v1; Q(4,5) = b_ * v1; Q(5,4) = b_ * v1; Q(5,5) = c_ * v1; // Z
    Q(6,6) = a_ * v2; Q(6,7) = b_ * v2; Q(7,6) = b_ * v2; Q(7,7) = c_ * v2; // Yaw
    return Q;
  };

  Eigen::MatrixXd Q1 = build_Q(v1_1, v2_1);
  Eigen::MatrixXd Q2 = build_Q(v1_2, v2_2);

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站特判 (对两个滤波器都应用)
  if (this->convergened() && this->name == ArmorName::outpost) {
    if (std::abs(this->ekf_1_.x[7]) > 2) this->ekf_1_.x[7] = this->ekf_1_.x[7] > 0 ? 2.51 : -2.51;
    if (std::abs(this->ekf_2_.x[7]) > 2) this->ekf_2_.x[7] = this->ekf_2_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_1_.predict(F, Q1, f);
  ekf_2_.predict(F, Q2, f);

  // 预测阶段结束后初步更新 combined_x_，以便其它函数在此间隙调用时不会出错
  combined_x_ = mix_states(ekf_1_.x, ekf_2_.x, mu_[0], mu_[1]);
}

void Target::update(const Armor & armor)
{
  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;
  
  if(name == ArmorName::outpost){
    double a = 0.1;
    tower_armor_h = a*armor.xyz_in_world[2] + (1-a)*last_tower_armor_h[id];
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr++;     
  }

  if (id != last_id) {
    is_switch_ = true;
    if(name == ArmorName::outpost){
      tower_armor_hs[last_id] = tower_armor_hs_datas[last_id] / (tower_armor_hs_datas_ptr + 1);
      tower_armor_hs_datas_ptr = 0;
      tower_armor_hs_datas[last_id] = 0;        
    }
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;    
  xyz_in_world = armor.xyz_in_world;

  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  auto r2_azimuth = 4e-3;
  auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  auto r2_d = log(std::abs(delta_angle) + 1) + 1;
  
  if(last_cam_is_short != cam_is_short){
    cam_is_switch_time_point = std::chrono::steady_clock::now();
    last_cam_is_short = cam_is_short;
    tools::logger()->info("[Target] last_cam_is_short != cam_is_short {}", (bool)(last_cam_is_short != cam_is_short));
  }
  auto now = std::chrono::steady_clock::now();
  double cam_is_switch_lter_dt = tools::delta_time(now, cam_is_switch_time_point);
  if(cam_is_switch_lter_dt < 0.7 && update_count_ > 50){
    r2_azimuth = 4e+4;
    r2_angle *= 300;
    r2_d *= 300;
  }
  
  Eigen::VectorXd R_dig{{r2_azimuth, r2_azimuth, r2_d, r2_angle}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  // --- IMM 步骤 3: 观测与似然更新 ---
  Eigen::MatrixXd H1 = h_jacobian(ekf_1_.x, id);
  Eigen::MatrixXd H2 = h_jacobian(ekf_2_.x, id);

  Eigen::Vector4d z_pred1 = h(ekf_1_.x);
  Eigen::Vector4d z_pred2 = h(ekf_2_.x);

  Eigen::VectorXd y1 = z_subtract(z, z_pred1);
  Eigen::VectorXd y2 = z_subtract(z, z_pred2);

  // 计算新息协方差 S
  Eigen::MatrixXd S1 = H1 * ekf_1_.P * H1.transpose() + R;
  Eigen::MatrixXd S2 = H2 * ekf_2_.P * H2.transpose() + R;

  // 计算多维高斯似然 (Likelihood) - 忽略常数项
  double L1 = std::exp(-0.5 * y1.transpose() * S1.inverse() * y1) / std::sqrt(S1.determinant());
  double L2 = std::exp(-0.5 * y2.transpose() * S2.inverse() * y2) / std::sqrt(S2.determinant());

  // 防止出现极小值导致除零
  L1 = std::max(L1, 1e-15);
  L2 = std::max(L2, 1e-15);

  // 更新模型概率并归一化
  mu_[0] = mu_[0] * L1;
  mu_[1] = mu_[1] * L2;
  mu_ /= mu_.sum(); 

  // 分别更新两个滤波器
  ekf_1_.update(z, H1, R, h, z_subtract);
  ekf_2_.update(z, H2, R, h, z_subtract);

  // --- IMM 步骤 4: 最终状态融合 ---
  combined_x_ = mix_states(ekf_1_.x, ekf_2_.x, mu_[0], mu_[1]);

  // 目标静止或匀速时，mu_[0]（平滑模型）应该接近 1.0；当敌方发生急停或转向时，mu_[1] 会瞬间飙升，这说明 IMM 工作正常
  tools::logger()->info("Mu: {:.2f}, {:.2f}", mu_[0], mu_[1]);
}

Eigen::VectorXd Target::ekf_x() const { return combined_x_; }

// IMM 不再暴露单个滤波器实例，此处注释掉
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_1_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(combined_x_[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(combined_x_, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = combined_x_[8] > 0.05 && combined_x_[8] < 0.5;
  auto l_ok = combined_x_[8] + combined_x_[9] > 0.05 && combined_x_[8] + combined_x_[9] < 0.5;
  if (r_ok && l_ok) return false;
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }
  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);

  double armor_z;
  if(name == ArmorName::outpost){
    double dz = tower_armor_hs[id] - tower_armor_hs[0];
    int dz_px = dz > 0 ? 1 : -1;
    int dz_mu;
    if(abs(dz) > 0.16){
      dz_mu = 2;
    }else if(abs(dz) < 0.16 && abs(dz) > 0.05){
      dz_mu = 1;
    }else if(abs(dz) < 0.05){
      dz_mu = 0;
    }
    armor_z = x[4] + TOWTER_ARMOR_DH * dz_px * dz_mu;
  }else{
    armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  }
  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);
  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  double dz_dh;
  if(this->name == ArmorName::outpost){
    double dz = tower_armor_hs[id] - tower_armor_hs[0];
    int dz_px = dz > 0 ? 1 : -1;
    int dz_mu;
    if(abs(dz) > 0.16){
      dz_mu = 2;
    }else if(abs(dz) < 0.16 && abs(dz) > 0.05){
      dz_mu = 1;
    }else if(abs(dz) < 0.05){
      dz_mu = 0;
    }
    dz_dh = dz_mu * dz_px;
  }else{
    dz_dh = (use_l_h) ? 1.0 : 0.0;
  }
  
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);

  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim