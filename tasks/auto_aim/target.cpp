#include "target.hpp"

#include <numeric>
#include <cmath>
#include <functional>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

constexpr double TOWTER_ARMOR_DH = 0.108; //前哨站两个装甲板之间的最短高低差m
constexpr double TOWER_ARMOR_DTB = 0.16;  //前哨装甲大跳变m
constexpr double TOWER_ARMOR_XTB = 0.05;  //前哨装甲小跳变m

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
  reject_count_(0), // [新增] 初始化拒收计数器
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0),
  is_rotation_cv_(false) // 默认初始为平移 CV 模型
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

  // 恢复 11 维：纯 CV 状态
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);
  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0;
  
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
: armor_num_(4),
  reject_count_(0), // [新增] 初始化拒收计数器
  is_rotation_cv_(false)
{
  // 恢复 11 维
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
  // ================= 平移/旋转 CV 滞回比较逻辑 =================
  double vyaw = std::abs(ekf_.x[7]);
  
  if (!is_rotation_cv_) {
    // 当前是平移 CV，如果角速度 >= 1.5 rad/s，切到旋转 CV
    if (vyaw >= 1.5) {
      is_rotation_cv_ = true;
    }
  } else {
    // 当前是旋转 CV，如果角速度 <= 0.5 rad/s，切回平移 CV
    if (vyaw <= 0.5) {
      is_rotation_cv_ = false;
    }
  }
  
  // 11维基础转移矩阵
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 5;     v2 = 0.05;
  } else {
    // ================= 核心：对两种 CV 分配不同的噪声 =================
    if (is_rotation_cv_) {
      // 旋转 CV：锁定车辆中心(v1极小)，紧盯小陀螺自转(v2极大)
      v1 = 5.0;   // 抑制 X/Y 轴的乱跳，认为车辆中心大致不动或缓慢匀速
      v2 = 0.1; // 紧随 Yaw 角的变化
    } else {
      // 平移 CV：跟踪车辆平移机动(v1大)，过滤 Yaw 的干扰(v2小)
      v1 = 100;  // 灵活跟踪平移
      v2 = 400;   // 认为车体朝向不会突变
    }
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
}

void Target::update(const std::vector<Armor> & armors)
{
  if (armors.empty()) return;

  int m = armors.size();
  int n = armor_num_; // EKF预测的ID数量 (通常为4)
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  // 1. 构建代价矩阵 (Cost Matrix)
  std::vector<std::vector<double>> cost_matrix(m, std::vector<double>(n, 0.0));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza_list[j].head(3));
      double yaw_error = tools::limit_rad(armors[i].ypr_in_world[0] - xyza_list[j][3]);
      double pitch_error = tools::limit_rad(armors[i].ypd_in_world[0] - ypd[0]);
      // 将角度偏差作为匹配代价
      cost_matrix[i][j] = std::abs(yaw_error) + std::abs(pitch_error);
    }
  }

  // 2. 使用全排列暴搜寻找全局最优分配（替代外部匈牙利算法库，N<=4时等效且更快）
  std::vector<int> best_assignment(m, -1);
  double min_total_cost = 1e9;
  
  std::vector<int> current_assignment(m, -1);
  std::vector<bool> used_id(n, false);

  std::function<void(int, double)> dfs = [&](int depth, double current_cost) {
    if (depth == m) {
      // 记录全局最小代价及其匹配方案
      if (current_cost < min_total_cost) {
        min_total_cost = current_cost;
        best_assignment = current_assignment;
      }
      return;
    }
    for (int j = 0; j < n; ++j) {
      if (!used_id[j]) {
        used_id[j] = true;
        current_assignment[depth] = j;
        dfs(depth + 1, current_cost + cost_matrix[depth][j]);
        used_id[j] = false;
      }
    }
  };
  
  dfs(0, 0.0);

  // 3. 根据最优匹配结果，依次对 EKF 进行更新
  for (int i = 0; i < m; ++i) {
    int id = best_assignment[i];
    const Armor & armor = armors[i];

    if (id != 0) jumped = true;
    
    // --- 延续原有的前哨站和记录逻辑 ---
    if(name == ArmorName::outpost){
      double a = 0.1;
      tower_armor_h = a*armor.xyz_in_world[2] + (1-a)*last_tower_armor_h[id];
      tower_armor_hs_datas[id] += tower_armor_h;
      last_tower_armor_h[id] = tower_armor_h;
      tower_armor_hs_datas_ptr[id]++;     
    }

    if (id != last_id) {
      is_switch_ = true;
      if(name == ArmorName::outpost){
        double a = 0.1;
        tower_armor_h = a*armor.xyz_in_world[2] + (1-a)*last_tower_armor_h[id];
        tower_armor_hs_datas[id] += tower_armor_h;
        last_tower_armor_h[id] = tower_armor_h;
        tower_armor_hs_datas_ptr[id]++;     
      }

      if(tower_armor_hs_datas[id] > 10000){
        tower_armor_hs_datas[id] = tower_armor_hs_datas[id] / (tower_armor_hs_datas_ptr[id] + 1);
        tower_armor_hs_datas[id] *=600;
        tower_armor_hs_datas_ptr[id] = 599;
      }
    } else {
      is_switch_ = false;
    }
      
    if (id != last_id) {
      is_switch_ = true;
      if(name == ArmorName::outpost){
        tower_armor_hs[last_id] = tower_armor_hs_datas[last_id] / (tower_armor_hs_datas_ptr[last_id] + 1);
      }
    } 

    last_id = id;
    update_count_++;    
    xyz_in_world = armor.xyz_in_world;
    // ----------------------------------

    // 将匹配好的装甲板依次送入 EKF 观测更新（卡尔曼可以安全地连续多次迭代更新）
    update_ypda(armor, id); 
  }
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  auto r2_azimuth = 4e-3;
  auto r2_pitch = 4e-3;
  // [修改] 提高角度观测的基础噪声，从 9e-2 提高到 0.15，降低滤波器对单帧跳动数据的敏感度
  auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2; 
  auto r2_d = log(std::abs(delta_angle) + 1) + 1;
  
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

  // ==================== [新增] 马氏距离门控与防死锁机制 ====================
  bool should_update = true;
  
  // 仅在 EKF 初始化完成并有了一定置信度后，才开启残差过滤，防止初始收敛困难
  if (update_count_ > 10) {
    Eigen::VectorXd z_pred = h(ekf_.x);
    Eigen::VectorXd y = z_subtract(z, z_pred); // 观测残差 Innovation
    
    // 计算新息协方差矩阵 S = H * P * H^T + R
    Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;
    
    // 计算马氏距离平方 D^2 = y^T * S^-1 * y
    double mahalanobis_sq = y.transpose() * S.inverse() * y;
    
    // 观测维度自由度为 4 (azimuth, pitch, d, angle)
    // 根据卡方分布表，自由度4，99%置信度对应的阈值为 13.277
    double chi_square_threshold = 13.277; 

    if (mahalanobis_sq > chi_square_threshold) {
      reject_count_++;
      // 防死锁：如果连续 8 帧观测数据都被判定为异常，说明很可能不是噪声，而是目标发生急停/反转
      if (reject_count_ > 8) {
        should_update = true; // 强行拉回
        reject_count_ = 0;
        
        // 放大协方差矩阵，强制 EKF 大幅降低对过往状态的自信，重新相信最新观测
        ekf_.P *= 10.0; 
      } else {
        should_update = false; // 抛弃异常帧，只使用 predict 预测的结果
      }
    } else {
      reject_count_ = 0; // 观测正常，重置拒收计数器
    }
  }

  // 根据判断结果决定是否进行卡尔曼更新
  if (should_update) {
    ekf_.update(z, H, R, h, z_subtract);
  }
  // =========================================================================
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
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
  
  // 恢复 11 维大小
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
