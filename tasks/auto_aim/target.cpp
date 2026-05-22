#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 物理常量定义
constexpr double TOWER_ARMOR_DH = 0.10;
constexpr double TOWER_ARMOR_DTB = 0.16;
constexpr double TOWER_ARMOR_XTB = 0.05;

namespace auto_aim
{

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
  // 【新增】分配单板CA滤波器数组空间
  armor_ca_kfs_.resize(armor_num_);

  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 根据当前装甲板位置和半径，反推旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  if(name == ArmorName::outpost){
    tower_armor_hs[0].first = true;
    tower_armor_hs[0].second = center_z;
  }

  cam_is_switch_time_point = std::chrono::steady_clock::time_point{};

  // ==========================================
  // EKF 11维状态向量定义:
  // [0]x, [1]vx, [2]y, [3]vy, [4]z, [5]vz, 
  // [6]yaw(偏航角), [7]vyaw(自转角速度), 
  // [8]r(基础半径), [9]r_(半径补偿量), [10]z_(高度补偿量)
  // ==========================================

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);
  double initial_dz = (name == ArmorName::outpost) ? TOWER_ARMOR_DH : 0.0;
  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, initial_dz;
  
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 10.0;
  P0.block(0, 0, 11, 11) = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h) 
: armor_num_(4)
{
  // 【新增】分配单板CA滤波器数组空间
  armor_ca_kfs_.resize(4);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);
  x0 << x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h;

  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 10.0;
  P0.block(0, 0, 11, 11) = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;
    v2 = 0.1;
  } else {
    v1 = 100;
    v2 = 400;
  }

  auto a_ = dt * dt * dt * dt / 4;
  auto b_ = dt * dt * dt / 2;
  auto c_ = dt * dt;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
  Q(0,0) = a_ * v1; Q(0,1) = b_ * v1; Q(1,0) = b_ * v1; Q(1,1) = c_ * v1; // X
  Q(2,2) = a_ * v1; Q(2,3) = b_ * v1; Q(3,2) = b_ * v1; Q(3,3) = c_ * v1; // Y
  Q(4,4) = a_ * v1; Q(4,5) = b_ * v1; Q(5,4) = b_ * v1; Q(5,5) = c_ * v1; // Z
  Q(6,6) = a_ * v2; Q(6,7) = b_ * v2; Q(7,6) = b_ * v2; Q(7,7) = c_ * v2; // Yaw

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (this->convergened() && this->name == ArmorName::outpost) {
    if (std::abs(this->ekf_.x[7]) > 2) this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);

  // 【新增】对已经初始化的各面单板独立CA模型进行预测
  for (auto & ca_kf : armor_ca_kfs_) {
    ca_kf.predict(dt);
  }
}

void Target::update(const Armor & armor)
{
  int id = 0;

  if (this->name == ArmorName::outpost) {
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

    this->ekf_x()(10) = TOWER_ARMOR_DH;

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

  } else {
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

    auto r2_azimuth = 4e-3;
    auto r2_pitch = 4e-3;
    auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
    auto r2_d = log(std::abs(delta_angle) + 1) + 1;
    
    if (last_cam_is_short != cam_is_short) {
      cam_is_switch_time_point = std::chrono::steady_clock::now();
      last_cam_is_short = cam_is_short;
    }
    auto now = std::chrono::steady_clock::now();
    double cam_is_switch_lter_dt = tools::delta_time(now, cam_is_switch_time_point);
    if (cam_is_switch_lter_dt < 0.7 && update_count_ > 50) {
      r2_azimuth = 4e+4;
      r2_angle *= 300;
      r2_d *= 300;
    }
    
    Eigen::VectorXd R_dig{{r2_azimuth, r2_pitch, r2_d, r2_angle}};
    Eigen::MatrixXd R = R_dig.asDiagonal();

    const Eigen::VectorXd & ypd = armor.ypd_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;
    Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

    int best_id = 0;
    double min_mahalanobis_dist = 1e10;
    std::vector<double> md_list(armor_num_, 1e10);

    for (int i = 0; i < armor_num_; i++) {
      Eigen::VectorXd xyz_pred = h_armor_xyz(ekf_.x, i);
      Eigen::VectorXd ypd_pred = tools::xyz2ypd(xyz_pred);
      auto angle_pred = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
      Eigen::VectorXd z_pred{{ypd_pred[0], ypd_pred[1], ypd_pred[2], angle_pred}};

      Eigen::VectorXd y = z - z_pred;
      y[0] = tools::limit_rad(y[0]);
      y[1] = tools::limit_rad(y[1]);
      y[3] = tools::limit_rad(y[3]);

      Eigen::MatrixXd H = h_jacobian(ekf_.x, i);
      Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;

      double mahalanobis_dist = y.transpose() * S.inverse() * y;
      md_list[i] = mahalanobis_dist;

      if (mahalanobis_dist < min_mahalanobis_dist) {
        min_mahalanobis_dist = mahalanobis_dist;
        best_id = i;
      }
    }

    id = best_id;
    double CHI_SQ_THRESHOLD = 9.488; 
    double HYSTERESIS_MARGIN = 5.0; 

    if (md_list[last_id] < CHI_SQ_THRESHOLD) {
      if (min_mahalanobis_dist > md_list[last_id] - HYSTERESIS_MARGIN) {
        id = last_id;
      }
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
    
    if (name == ArmorName::outpost) {
      if (tower_armor_hs_datas_ptr[last_id] > 0) {
        tower_armor_hs[last_id].first = true;
        tower_armor_hs[last_id].second = tower_armor_hs_datas[last_id] / tower_armor_hs_datas_ptr[last_id];
      }
    }
  } else {
    is_switch_ = false;
  }

  if(name == ArmorName::outpost){
    double a = 0.1;
    tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
    
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr[id]++;     

    if(tower_armor_hs_datas[id] > 10000){
      tower_armor_hs_datas[id] = (tower_armor_hs_datas[id] / tower_armor_hs_datas_ptr[id]) * 600;
      tower_armor_hs_datas_ptr[id] = 600;
    }
  }

  last_id = id;
  update_count_++;    
  xyz_in_world = armor.xyz_in_world;

  // 【新增】更新当前关联上的独立装甲板的 CA Filter
  if (id >= 0 && id < (int)armor_ca_kfs_.size()) {
      if (!armor_ca_kfs_[id].is_initialized) {
          armor_ca_kfs_[id].init(armor.xyz_in_world);
      } else {
          armor_ca_kfs_[id].update(armor.xyz_in_world);
      }
  }

  // 调用 EKF 更新观测值
  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  // 用距离判断观测数据，距离越远观测越不可信
  double distance = std::abs(armor.ypd_in_world[2]);
  double dist_penalty = distance * distance * 0.05;

  auto r2_azimuth = 4e-3 + distance * 1e-3;
  auto r2_pitch   = 4e-3 + distance * 1e-3;
  auto r2_angle   = log(distance + 1) / 200 + 9e-2 + dist_penalty * 0.1;
  auto r2_d       = log(std::abs(delta_angle) + 1) + 1 + dist_penalty;

  // auto r2_azimuth = 4e-3;
  // auto r2_pitch = 4e-3;
  // auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  // auto r2_d = log(std::abs(delta_angle) + 1) + 1;
  
  if(last_cam_is_short != cam_is_short){
    cam_is_switch_time_point = std::chrono::steady_clock::now();
    last_cam_is_short = cam_is_short;
  }
  auto now = std::chrono::steady_clock::now();
  double cam_is_switch_lter_dt = tools::delta_time(now, cam_is_switch_time_point);
  if(cam_is_switch_lter_dt < 0.7 && update_count_ > 50){
    r2_azimuth = 4e+4;
    r2_angle *= 300;
    r2_d *= 300;
  }
  
  Eigen::VectorXd R_dig{{r2_azimuth, r2_pitch, r2_d, r2_angle}};
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

  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  
  // 【优化】使用基于马尔可夫指数分布的机动频率连续融合策略
  double vyaw = ekf_.x[7]; // 当前整车自旋角速度
  constexpr double OMEGA_THRESH = 1.5; // 机动频率融合时间常数/阈值 (rad/s)
  
  // 核心逻辑：旋转越快，weight_CV 渐进趋近于1 (偏向整车CV)；静止/平移时，趋近于0 (偏向单板CA)
  // 指数函数保证了权重变化的一阶导数连续，避免云台追踪时出现“折点”震荡
  double weight_CV = 1.0 - std::exp(-std::abs(vyaw) / OMEGA_THRESH);

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    
    // 由全车CV模型反推的单块装甲板预测位置
    Eigen::Vector3d cv_xyz = h_armor_xyz(ekf_.x, i);
    Eigen::Vector3d final_xyz = cv_xyz;

    // 如果当前面的单板CA模型已经经过初始化并且有效迭代了数次，执行平滑融合
    if (i < (int)armor_ca_kfs_.size() && armor_ca_kfs_[i].is_initialized && armor_ca_kfs_[i].update_count > 3) {
        Eigen::Vector3d ca_xyz = armor_ca_kfs_[i].kf.x.head(3);
        // 使用连续的 weight_CV 进行互补融合
        final_xyz = weight_CV * cv_xyz + (1.0 - weight_CV) * ca_xyz;
    }

    _armor_xyza_list.push_back({final_xyz[0], final_xyz[1], final_xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
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
      double dz = tower_armor_hs[id].second - tower_armor_hs[0].second;
      int dz_px = dz > 0 ? 1 : -1;
      int dz_mu;
      
      if (std::abs(dz) > TOWER_ARMOR_DTB) {
        dz_mu = 2; 
      } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
        dz_mu = 1;
      } else {
        dz_mu = 0; 
      }
      
      armor_z = x[4] + x[10] * dz_px * dz_mu; 
    } else {
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
    double dz = tower_armor_hs[id].second - tower_armor_hs[0].second;
    int dz_px = dz > 0 ? 1 : -1;
    int dz_mu;
    
    if (std::abs(dz) > TOWER_ARMOR_DTB) {
      dz_mu = 2;
    } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
      dz_mu = 1;
    } else {
      dz_mu = 0;
    }
    dz_dh = dz_mu * dz_px;
  }else{
    dz_dh = (use_l_h) ? 1.0 : 0.0;
  }
  
  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
  H_armor_xyza(0, 0) = 1; H_armor_xyza(0, 6) = dx_da; H_armor_xyza(0, 8) = dx_dr; H_armor_xyza(0, 9) = dx_dl;
  H_armor_xyza(1, 2) = 1; H_armor_xyza(1, 6) = dy_da; H_armor_xyza(1, 8) = dy_dr; H_armor_xyza(1, 9) = dy_dl;
  H_armor_xyza(2, 4) = 1; H_armor_xyza(2, 10) = dz_dh;
  H_armor_xyza(3, 6) = 1;

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