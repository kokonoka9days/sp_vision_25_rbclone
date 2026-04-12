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
  switch_count_(0),
  motion_state_(MotionState::TRANSLATION) // 默认初始为平移模型
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
  motion_state_(MotionState::TRANSLATION)
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
  double vyaw = std::abs(ekf_.x[7]);
  // 计算 X 和 Y 方向的合成线速度 (m/s)
  double v_linear = std::hypot(ekf_.x[1], ekf_.x[3]); 
  
  // 状态机滞回阈值配置 (需要根据实际底盘性能微调)
  const double OMEGA_HIGH = 3;   // 进入旋转的角速度阈值 (rad/s)
  const double OMEGA_LOW = 1.5;    // 退出旋转的角速度阈值 (rad/s)
  const double V_HIGH = 0.6;       // 进入平移旋转的线速度阈值 (m/s)
  const double V_LOW = 0.3;        // 退出平移旋转的线速度阈值 (m/s)

  // ================= 状态转移逻辑 =================
  switch (motion_state_) {
    case MotionState::TRANSLATION:
      if (vyaw > OMEGA_HIGH) {
        if (v_linear > V_HIGH) motion_state_ = MotionState::TRANSLATION_ROTATION;
        else motion_state_ = MotionState::IN_PLACE_ROTATION;
      }
      break;

    case MotionState::IN_PLACE_ROTATION:
      if (vyaw < OMEGA_LOW) {
        motion_state_ = MotionState::TRANSLATION;
      } else if (v_linear > V_HIGH) {
        motion_state_ = MotionState::TRANSLATION_ROTATION;
      }
      break;

    case MotionState::TRANSLATION_ROTATION:
      if (vyaw < OMEGA_LOW) {
        motion_state_ = MotionState::TRANSLATION;
      } else if (v_linear < V_LOW) {
        motion_state_ = MotionState::IN_PLACE_ROTATION;
      }
      break;
  }
  
  // 11维基础转移矩阵
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 5;     v2 = 0.05;
  } else {
    // ================= 核心：利用枚举分配不同的噪声 =================
    switch (motion_state_) {
      case MotionState::TRANSLATION:
        // 纯平移：灵活跟踪平移，过滤 Yaw 的干扰
        v1 = 100;
        v2 = 20;
        break;

      case MotionState::IN_PLACE_ROTATION:
        // 原地旋转：抑制 X/Y 轴的乱跳，紧随 Yaw 角的变化
        v1 = 1;    
        v2 = 0.1;
        break;

      case MotionState::TRANSLATION_ROTATION:
        // 平移 + 旋转：车辆同时剧烈平移和自转，赋予全部坐标轴较高的灵活性
        v1 = 100;
        v2 = 400;
        break;
    }
  }

  // tools::logger()->info("[Target] 当前目标ekf模式: {}", static_cast<int>(motion_state_));

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

void Target::update(const Armor & armor)
{
  // ================= 1. 提前构造测量噪声矩阵 R (复用 update_ypda 中的逻辑) =================
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

  // ================= 2. 构造当前实际的观测向量 z =================
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  // ================= 3. 基于马氏距离寻找最佳匹配装甲板 =================
  int best_id = 0;
  double min_mahalanobis_dist = 1e10;
  std::vector<double> md_list(armor_num_, 1e10);

  for (int i = 0; i < armor_num_; i++) {
    // 3.1 计算预测观测值 h(x)
    Eigen::VectorXd xyz_pred = h_armor_xyz(ekf_.x, i);
    Eigen::VectorXd ypd_pred = tools::xyz2ypd(xyz_pred);
    auto angle_pred = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::VectorXd z_pred{{ypd_pred[0], ypd_pred[1], ypd_pred[2], angle_pred}};

    // 3.2 计算残差 y = z - h(x)
    Eigen::VectorXd y = z - z_pred;
    y[0] = tools::limit_rad(y[0]);
    y[1] = tools::limit_rad(y[1]);
    y[3] = tools::limit_rad(y[3]);

    // 3.3 获取雅克比矩阵 H
    Eigen::MatrixXd H = h_jacobian(ekf_.x, i);

    // 3.4 计算新息协方差矩阵 S = H * P * H^T + R
    // 注意：这要求 ekf_.P 是可访问的。若在你的 ExtendedKalmanFilter 类中 P 是 public 的，可以直接使用
    Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;

    // 3.5 计算马氏距离平方 D_M^2 = y^T * S^-1 * y
    double mahalanobis_dist = y.transpose() * S.inverse() * y;
    md_list[i] = mahalanobis_dist;

    if (mahalanobis_dist < min_mahalanobis_dist) {
      min_mahalanobis_dist = mahalanobis_dist;
      best_id = i;
    }
  }

  // ================= 4. 迟滞防抖动（Hysteresis）限制跳变 =================
  int id = best_id;
  // 4自由度的卡方分布 95% 置信度阈值约为 9.488，99% 为 13.277
  double CHI_SQ_THRESHOLD = 9.488; 
  double HYSTERESIS_MARGIN = 5.0; // 迟滞裕度，用于防止在两块装甲板马氏距离接近时反复横跳

  // 如果保持上一次的 ID 仍然在一个合理且可信的范围内
  if (md_list[last_id] < CHI_SQ_THRESHOLD) {
    // 除非新的匹配度显著地好于原来的匹配度（好过迟滞裕度），否则拒接跳变
    if (min_mahalanobis_dist > md_list[last_id] - HYSTERESIS_MARGIN) {
      id = last_id;
    }
  }

  if (id != 0) jumped = true;
  
  // ================= 5. 更新前哨站和切换状态（保留原逻辑）=================
  if(name == ArmorName::outpost){
    double a = 0.1;
    tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr[id]++;     
  }

  if (id != last_id) {
    is_switch_ = true;
    if(name == ArmorName::outpost){
      double a = 0.1;
      tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
      tower_armor_hs_datas[id] += tower_armor_h;
      last_tower_armor_h[id] = tower_armor_h;
      tower_armor_hs_datas_ptr[id]++;     
    }

    if(tower_armor_hs_datas[id] > 10000){
      tower_armor_hs_datas[id] = tower_armor_hs_datas[id] / (tower_armor_hs_datas_ptr[id] + 1);
      tower_armor_hs_datas[id] *= 600;
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

  // 最后调用更新滤波器
  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  auto r2_azimuth = 4e-3;
  auto r2_pitch = 4e-3;
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

  ekf_.update(z, H, R, h, z_subtract);
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