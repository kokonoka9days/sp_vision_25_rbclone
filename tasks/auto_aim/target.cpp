#include "target.hpp"

#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 物理常量定义
constexpr double TOWER_ARMOR_DH = 0.10;  // 前哨站两个装甲板之间的标准高低差(m)
constexpr double TOWER_ARMOR_DTB = 0.16;  // 前哨装甲大跳变阈值(m)
constexpr double TOWER_ARMOR_XTB = 0.05;  // 前哨装甲小跳变阈值(m)

namespace auto_aim
{

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  color(armor.color),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  nominal_radius_(radius),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 根据当前装甲板位置和半径，反推旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  if(name == ArmorName::outpost){
    tower_armor_hs[0].first = true;       // 标记 0 号位已成功初始化
    tower_armor_hs[0].second = center_z;  // 记录真实高度
  }

  cam_is_switch_time_point = std::chrono::steady_clock::time_point{};

  // ==========================================
  // EKF 11维状态向量定义:
  // [0]x, [1]vx, [2]y, [3]vy, [4]z, [5]vz, 
  // [6]yaw(偏航角), [7]vyaw(自转角速度), 
  // [8]r(基础半径), [9]r_(半径补偿量), [10]z_(高度补偿量)
  // ==========================================
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);

  // 如果是前哨站，将 z_ (x[10]) 的初始值设为物理理论值
  double initial_dz = (name == ArmorName::outpost) ? TOWER_ARMOR_DH : 0.0;

  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, initial_dz;
  
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 10.0;
  P0.block(0, 0, 11, 11) = P0_dig.asDiagonal();

  // 自定义状态加法，确保角度(Yaw)在 -PI 到 PI 之间
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  observed_center_ = {center_x, center_y, center_z};
  cv_position_at_observation_ = observed_center_;
  observation_time_ = t;
  init_ca_filter(observed_center_, Eigen::Vector3d::Zero());
  record_center_observation(t, observed_center_);
}

// 供手动初始化使用的构造函数
Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4), nominal_radius_(radius), t_(std::chrono::steady_clock::time_point{})
{
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
  observed_center_ = {x, 0, 0};
  cv_position_at_observation_ = observed_center_;
  observation_time_ = t_;
  init_ca_filter(observed_center_, Eigen::Vector3d::Zero());
  record_center_observation(t_, observed_center_);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (!std::isfinite(dt) || std::abs(dt) < 1e-9) return;
  prediction_age_ += dt;

  // CV model for the complete vehicle state.
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
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

  // 前哨站收敛后限制最大转速防飞
  if (this->convergened() && this->name == ArmorName::outpost) {
    if (std::abs(this->ekf_.x[7]) > 2) this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);

  if (!ca_ekf_init_) return;
  // A measurement update may request an implausible acceleration. Bound it before the state
  // transition so the rejected impulse never contributes to the next position prediction.
  clamp_ca_state();

  // Standard CA transition paired with its matching continuous white-jerk Q.
  Eigen::MatrixXd F_ca = Eigen::MatrixXd::Identity(9, 9);
  for (int axis = 0; axis < 3; ++axis) {
    const int j = axis * 3;
    F_ca(j, j + 1) = dt;
    F_ca(j, j + 2) = 0.5 * dt * dt;
    F_ca(j + 1, j + 2) = dt;
  }

  const double T = std::abs(dt);
  const double direction = dt >= 0 ? 1.0 : -1.0;
  const double T2 = T * T;
  const double T3 = T2 * T;
  const double T4 = T3 * T;
  const double T5 = T4 * T;
  Eigen::MatrixXd Q_ca = Eigen::MatrixXd::Zero(9, 9);
  for (int axis = 0; axis < 3; ++axis) {
    const int j = axis * 3;
    const double q = axis < 2 ? 200.0 : 40.0;
    Q_ca(j, j) = q * T5 / 20.0;
    Q_ca(j, j + 1) = Q_ca(j + 1, j) = direction * q * T4 / 8.0;
    Q_ca(j, j + 2) = Q_ca(j + 2, j) = q * T3 / 6.0;
    Q_ca(j + 1, j + 1) = q * T3 / 3.0;
    Q_ca(j + 1, j + 2) = Q_ca(j + 2, j + 1) = direction * q * T2 / 2.0;
    Q_ca(j + 2, j + 2) = q * T;
  }
  ca_ekf_.predict(F_ca, Q_ca);
}

void Target::init_ca_filter(
  const Eigen::Vector3d & center, const Eigen::Vector3d & velocity)
{
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(9);
  x0 << center.x(), velocity.x(), 0.0, center.y(), velocity.y(), 0.0, center.z(), velocity.z(),
    0.0;

  Eigen::VectorXd P0_diagonal(9);
  P0_diagonal << 0.04, 9.0, 100.0, 0.04, 9.0, 100.0, 0.09, 9.0, 64.0;
  ca_ekf_ = tools::ExtendedKalmanFilter(x0, P0_diagonal.asDiagonal());
  ca_position_at_observation_ = center;
  ca_ekf_init_ = true;
  ca_update_count_ = 0;
}

void Target::clamp_ca_state()
{
  if (!ca_ekf_init_ || ca_ekf_.x.size() != 9) return;
  constexpr double MAX_ACCEL_XY = 15.0;
  constexpr double MAX_ACCEL_Z = 8.0;
  ca_ekf_.x[2] = std::clamp(ca_ekf_.x[2], -MAX_ACCEL_XY, MAX_ACCEL_XY);
  ca_ekf_.x[5] = std::clamp(ca_ekf_.x[5], -MAX_ACCEL_XY, MAX_ACCEL_XY);
  ca_ekf_.x[8] = std::clamp(ca_ekf_.x[8], -MAX_ACCEL_Z, MAX_ACCEL_Z);
}

void Target::record_center_observation(
  std::chrono::steady_clock::time_point time, const Eigen::Vector3d & center)
{
  if (!center.allFinite()) return;
  if (!center_history_.empty() && center_history_.back().time == time) {
    center_history_.back().center = center;
  } else {
    center_history_.push_back({time, center});
  }

  constexpr double HISTORY_SECONDS = 0.5;
  while (
    center_history_.size() > 2 &&
    tools::delta_time(time, center_history_.front().time) > HISTORY_SECONDS) {
    center_history_.pop_front();
  }
}

Eigen::Vector3d Target::align_with_bearing_prediction(const Eigen::Vector3d & center) const
{
  const double range = center.norm();
  if (!std::isfinite(range) || range < 1e-6 || center_history_.size() < 4) return center;

  constexpr double FIT_WINDOW = 0.10;
  const auto latest_time = center_history_.back().time;
  std::vector<Eigen::Vector3d> samples;
  samples.reserve(center_history_.size());
  double previous_raw_yaw = 0.0;
  double previous_unwrapped_yaw = 0.0;
  bool first = true;
  for (const auto & observation : center_history_) {
    const double dt = tools::delta_time(observation.time, latest_time);
    if (dt < -FIT_WINDOW || observation.center.norm() < 1e-6) continue;
    const double raw_yaw = std::atan2(observation.center.y(), observation.center.x());
    const double pitch = std::atan2(
      observation.center.z(), std::hypot(observation.center.x(), observation.center.y()));
    double unwrapped_yaw = raw_yaw;
    if (!first) {
      unwrapped_yaw = previous_unwrapped_yaw + tools::limit_rad(raw_yaw - previous_raw_yaw);
    }
    samples.emplace_back(dt, unwrapped_yaw, pitch);
    previous_raw_yaw = raw_yaw;
    previous_unwrapped_yaw = unwrapped_yaw;
    first = false;
  }
  if (samples.size() < 4) return center;

  std::vector<double> weights(samples.size(), 1.0);
  Eigen::Matrix2d coefficients = Eigen::Matrix2d::Zero();
  bool fit_valid = false;
  for (int iteration = 0; iteration < 4; ++iteration) {
    Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d rhs = Eigen::Matrix2d::Zero();
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const Eigen::Vector2d basis(1.0, samples[i].x());
      const Eigen::Vector2d angles(samples[i].y(), samples[i].z());
      normal.noalias() += weights[i] * basis * basis.transpose();
      rhs.noalias() += weights[i] * basis * angles.transpose();
    }
    const Eigen::LDLT<Eigen::Matrix2d> ldlt(normal);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) break;
    coefficients = ldlt.solve(rhs);
    if (!coefficients.allFinite()) break;
    fit_valid = true;

    std::vector<double> residuals;
    residuals.reserve(samples.size());
    for (const auto & sample : samples) {
      const Eigen::Vector2d basis(1.0, sample.x());
      const Eigen::Vector2d angles(sample.y(), sample.z());
      residuals.push_back((coefficients.transpose() * basis - angles).norm());
    }
    std::vector<double> sorted_residuals = residuals;
    std::sort(sorted_residuals.begin(), sorted_residuals.end());
    const double scale = 1.4826 * sorted_residuals[sorted_residuals.size() / 2] + 1e-7;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
      const double u = residuals[i] / (3.0 * scale);
      if (u >= 1.0) {
        weights[i] = 0.02;
      } else {
        const double one_minus_square = 1.0 - u * u;
        weights[i] = one_minus_square * one_minus_square;
      }
    }
  }
  if (!fit_valid) return center;

  const double max_bearing_rate = 0.5 + 8.0 / std::max(range, 1.0);
  coefficients(1, 0) =
    std::clamp(coefficients(1, 0), -max_bearing_rate, max_bearing_rate);
  coefficients(1, 1) =
    std::clamp(coefficients(1, 1), -max_bearing_rate, max_bearing_rate);
  const Eigen::Vector2d predicted_angles =
    coefficients.transpose() * Eigen::Vector2d(1.0, prediction_age_);

  const double center_yaw = std::atan2(center.y(), center.x());
  const double center_pitch = std::atan2(center.z(), std::hypot(center.x(), center.y()));
  const double abs_age = std::abs(prediction_age_);
  const double bearing_weight = std::clamp((0.25 - abs_age) / 0.15, 0.0, 1.0);
  const double yaw = center_yaw +
                     bearing_weight * tools::limit_rad(predicted_angles.x() - center_yaw);
  const double pitch =
    center_pitch + bearing_weight * (predicted_angles.y() - center_pitch);
  const double horizontal_range = range * std::cos(pitch);
  return {
    horizontal_range * std::cos(yaw), horizontal_range * std::sin(yaw),
    range * std::sin(pitch)};
}

Eigen::Vector3d Target::center_from_armor(const Armor & armor, int id) const
{
  const bool use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
  const double radius = use_l_h ? ekf_.x[8] + ekf_.x[9] : ekf_.x[8];
  const double armor_yaw = armor.ypr_in_world[0];

  double center_z = armor.xyz_in_world[2];
  if (name == ArmorName::outpost) {
    const double dz = tower_armor_hs[id].second - tower_armor_hs[0].second;
    const double direction = dz >= 0.0 ? 1.0 : -1.0;
    const int steps = std::abs(dz) > TOWER_ARMOR_DTB ? 2 : std::abs(dz) > TOWER_ARMOR_XTB ? 1 : 0;
    center_z -= ekf_.x[10] * direction * steps;
  } else if (use_l_h) {
    center_z -= ekf_.x[10];
  }

  return {
    armor.xyz_in_world[0] + radius * std::cos(armor_yaw),
    armor.xyz_in_world[1] + radius * std::sin(armor_yaw), center_z};
}

bool Target::update(const Armor & armor)
{
  int id = 0;

  if (this->name == ArmorName::outpost) {
    // 【策略 A：前哨站专用】
    // 纯几何匹配(距离+复合角度)，绕开因高度阶梯跳变导致 EKF 协方差波动的干扰
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

    ekf_.x[10] = TOWER_ARMOR_DH;

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      xyza_i_list.push_back({xyza_list[i], i});
    }

    // 按距离(ypd[2])由近及远排序
    std::sort(
      xyza_i_list.begin(), xyza_i_list.end(),
      [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
        Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
        Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
        return ypd1[2] < ypd2[2];
      });

    // 只取最近的3个装甲板验证角度匹配度
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
    // 【策略 B：其他兵种通用】
    // 马氏距离匹配 + 迟滞防抖，有效应对平移带来的透视形变
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

    auto r2_azimuth = 4e-3;
    auto r2_pitch = 4e-3;
    auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
    auto r2_d = log(std::abs(delta_angle) + 1) + 1;
    
    // 处理镜头长短焦切换时的噪声激增
    if (last_cam_is_short != cam_is_short) {
      cam_is_switch_time_point = std::chrono::steady_clock::now();
      last_cam_is_short = cam_is_short;
    }
    if(last_cam_is_short){
      // tools::logger()->info("[Target] last_cam_is_short");
      
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

    // 迟滞防抖动（Hysteresis）机制：倾向于保持上一次匹配的ID
    id = best_id;
    double CHI_SQ_THRESHOLD = 9.488; 
    double HYSTERESIS_MARGIN = 5.0; 

    if (md_list[last_id] < CHI_SQ_THRESHOLD) {
      if (min_mahalanobis_dist > md_list[last_id] - HYSTERESIS_MARGIN) {
        id = last_id;
      }
    }
  }

  const Eigen::Vector3d center_measurement = center_from_armor(armor, id);
  if (!center_measurement.allFinite()) return false;
  if (update_count_ >= 5) {
    const double observation_dt = std::max(tools::delta_time(t_, observation_time_), 0.0);
    constexpr double MAX_TRANSLATION_SPEED = 8.0;
    const Eigen::Vector3d predicted_center = fused_center();
    const Eigen::Vector3d line_of_sight = predicted_center.normalized();
    const Eigen::Vector3d innovation = center_measurement - predicted_center;
    const double radial_innovation = std::abs(innovation.dot(line_of_sight));
    const double tangential_innovation =
      (innovation - innovation.dot(line_of_sight) * line_of_sight).norm();
    const double motion_allowance = MAX_TRANSLATION_SPEED * observation_dt;
    constexpr double TANGENTIAL_PNP_MARGIN = 0.10;
    constexpr double RADIAL_PNP_MARGIN = 0.80;
    if (tangential_innovation > TANGENTIAL_PNP_MARGIN + motion_allowance) return false;

    if (radial_innovation > RADIAL_PNP_MARGIN + motion_allowance) return false;
  }

  if (id != 0) jumped = true;

  // 检测换板事件
  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
    
    // 换板时，将上一块装甲板的历史累加数据计算为平均高度锚点
    if (name == ArmorName::outpost) {
      if (tower_armor_hs_datas_ptr[last_id] > 0) {
        tower_armor_hs[last_id].first = true; // 标记该装甲板已有有效的历史数据
        tower_armor_hs[last_id].second = tower_armor_hs_datas[last_id] / tower_armor_hs_datas_ptr[last_id];
      }
    }
  } else {
    is_switch_ = false;
  }

  // 累加当前块装甲板的高度特征
  if(name == ArmorName::outpost){
    double a = 0.1; // 互补滤波系数
    tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
    
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr[id]++;     

    // 历史高度数据保护机制，防止长时间追踪导致累加溢出
    if(tower_armor_hs_datas[id] > 10000){
      tower_armor_hs_datas[id] = (tower_armor_hs_datas[id] / tower_armor_hs_datas_ptr[id]) * 600;
      tower_armor_hs_datas_ptr[id] = 600;
    }
  }

  const Eigen::Vector3d cv_prior = cv_center();
  const Eigen::Vector3d ca_prior =
    ca_ekf_init_ ? Eigen::Vector3d(ca_ekf_.x[0], ca_ekf_.x[3], ca_ekf_.x[6]) : cv_prior;

  // Compare both models only against a measurement neither model has consumed yet.
  if (ca_update_count_ >= 3 && center_measurement.allFinite()) {
    const double cv_error = (cv_prior.head<2>() - center_measurement.head<2>()).squaredNorm();
    const double ca_error = (ca_prior.head<2>() - center_measurement.head<2>()).squaredNorm();
    if (cv_error < 1.0 && ca_error < 1.0) {
      constexpr double ERROR_EMA_KEEP = 0.90;
      if (!model_error_initialized_) {
        cv_error_ema_ = cv_error;
        ca_error_ema_ = ca_error;
        model_error_initialized_ = true;
      } else {
        cv_error_ema_ = ERROR_EMA_KEEP * cv_error_ema_ + (1.0 - ERROR_EMA_KEEP) * cv_error;
        ca_error_ema_ = ERROR_EMA_KEEP * ca_error_ema_ + (1.0 - ERROR_EMA_KEEP) * ca_error;
      }
    }
  }

  last_id = id;
  update_count_++;
  xyz_in_world = armor.xyz_in_world;
  observed_center_ = center_measurement;
  observation_time_ = t_;
  prediction_age_ = 0.0;

  update_ypda(armor, id);

  if (!ca_ekf_init_) {
    init_ca_filter(center_measurement, {ekf_.x[1], ekf_.x[3], ekf_.x[5]});
  } else {
    Eigen::MatrixXd H_ca = Eigen::MatrixXd::Zero(3, 9);
    H_ca(0, 0) = 1.0;
    H_ca(1, 3) = 1.0;
    H_ca(2, 6) = 1.0;

    const double distance = armor.xyz_in_world.norm();
    const double sigma_tangential = std::clamp(0.025 + 0.004 * distance, 0.04, 0.08);
    const double sigma_radial = std::clamp(0.12 + 0.03 * distance, 0.18, 0.45);
    const Eigen::Vector3d line_of_sight = center_measurement.normalized();
    Eigen::Matrix3d R_ca =
      sigma_tangential * sigma_tangential * Eigen::Matrix3d::Identity();
    R_ca.noalias() +=
      (sigma_radial * sigma_radial - sigma_tangential * sigma_tangential) *
      line_of_sight * line_of_sight.transpose();

    const Eigen::Vector3d residual = center_measurement - ca_prior;
    const Eigen::Matrix3d S = H_ca * ca_ekf_.P * H_ca.transpose() + R_ca;
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
    const double nis = ldlt.info() == Eigen::Success
                         ? residual.dot(ldlt.solve(residual))
                         : std::numeric_limits<double>::infinity();

    if (!std::isfinite(nis) || residual.norm() > 1.5) {
      init_ca_filter(center_measurement, {ekf_.x[1], ekf_.x[3], ekf_.x[5]});
      w_cv_ = 1.0;
      model_error_initialized_ = false;
    } else {
      // Inflate R instead of letting one PnP outlier become a false acceleration command.
      if (nis > 11.345) R_ca *= std::clamp(nis / 7.815, 1.0, 50.0);
      ca_ekf_.update(center_measurement, H_ca, R_ca);
      if (ca_ekf_.data.at("filter_update_rejected") < 0.5) ca_update_count_++;
    }
  }
  clamp_ca_state();

  if (name == ArmorName::outpost || ca_update_count_ < 8 || !model_error_initialized_) {
    w_cv_ = 1.0;
  } else {
    constexpr double ERROR_FLOOR = 0.0025;
    const double relative_ca = (cv_error_ema_ + ERROR_FLOOR) /
                               (cv_error_ema_ + ca_error_ema_ + 2.0 * ERROR_FLOOR);
    const double acceleration = std::hypot(ca_ekf_.x[2], ca_ekf_.x[5]);
    const double maneuver_evidence = std::clamp((acceleration - 0.4) / 2.5, 0.0, 1.0);
    const double ca_weight =
      std::clamp(relative_ca * (0.25 + 0.75 * maneuver_evidence), 0.05, 0.85);
    const double target_w_cv = 1.0 - ca_weight;
    w_cv_ = 0.8 * w_cv_ + 0.2 * target_w_cv;
  }

  // Anchor each model at the measurement it just consumed. Future output then uses the
  // model's displacement, without carrying its steady-state position lag into aiming.
  cv_position_at_observation_ = cv_center();
  ca_position_at_observation_ = {ca_ekf_.x[0], ca_ekf_.x[3], ca_ekf_.x[6]};
  record_center_observation(t_, center_measurement);
  return true;
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

  // 预测观测函数 h(x)
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 自定义减法（处理角度越界）
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

  const double min_radius = 0.60 * nominal_radius_;
  const double max_radius = 1.40 * nominal_radius_;
  ekf_.x[8] = std::clamp(ekf_.x[8], min_radius, max_radius);
  if (armor_num_ == 4) {
    const double long_radius = std::clamp(
      ekf_.x[8] + ekf_.x[9], 0.60 * nominal_radius_, 2.00 * nominal_radius_);
    ekf_.x[9] = long_radius - ekf_.x[8];
  }
}

// 获取 EKF 状态向量
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

// 获取滤波器常引用
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

Eigen::Vector3d Target::cv_center() const { return {ekf_.x[0], ekf_.x[2], ekf_.x[4]}; }

Eigen::Vector3d Target::fused_center() const
{
  const auto anchor_model = [&](
                              const Eigen::Vector3d & model_center,
                              const Eigen::Vector3d & model_at_observation) -> Eigen::Vector3d {
    const double observed_range = observed_center_.norm();
    if (!std::isfinite(observed_range) || observed_range < 1e-6) return model_center;
    const Eigen::Vector3d bias = observed_center_ - model_at_observation;
    const Eigen::Vector3d line_of_sight = observed_center_ / observed_range;
    const Eigen::Vector3d radial_bias = bias.dot(line_of_sight) * line_of_sight;
    const Eigen::Vector3d tangential_bias = bias - radial_bias;
    const double anchor_scale = std::exp(-std::abs(prediction_age_) / 0.05);
    return (model_center + anchor_scale * (tangential_bias + 0.15 * radial_bias)).eval();
  };

  const Eigen::Vector3d center_cv = anchor_model(cv_center(), cv_position_at_observation_);
  if (name == ArmorName::outpost || !ca_ekf_init_) {
    return align_with_bearing_prediction(center_cv);
  }

  const Eigen::Vector3d raw_center_ca(ca_ekf_.x[0], ca_ekf_.x[3], ca_ekf_.x[6]);
  const Eigen::Vector3d center_ca = anchor_model(raw_center_ca, ca_position_at_observation_);
  if (ca_update_count_ < 8 || !model_error_initialized_) {
    // During initialization or bounded reacquisition the CV state can be reset repeatedly while
    // the CA state still carries the local translation. Keep the output continuous at age zero,
    // then hand the short future prediction to CA instead of treating an unready weight as CV=1.
    const double ca_prediction_weight =
      std::clamp(std::abs(prediction_age_) / 0.03, 0.0, 1.0);
    return ca_prediction_weight * raw_center_ca +
           (1.0 - ca_prediction_weight) * observed_center_;
  }

  constexpr double FULL_CV_HORIZON = 0.20;
  const double horizon_scale =
    std::clamp(std::abs(prediction_age_) / FULL_CV_HORIZON, 0.05, 1.0);
  const double effective_w_cv = w_cv_ * horizon_scale;
  const Eigen::Vector3d bearing_enhanced = align_with_bearing_prediction(
    effective_w_cv * center_cv + (1.0 - effective_w_cv) * center_ca);

  // Close to an observation the anchor prevents a visible jump. By one 40 ms prediction
  // interval, the lower-noise CA state supplies most of the position while the robust bearing
  // fit still corrects the following direction.
  const double abs_age = std::abs(prediction_age_);
  const double short_ca_weight = 0.70 * std::clamp(abs_age / 0.04, 0.0, 1.0) *
                                 std::clamp((0.25 - abs_age) / 0.15, 0.0, 1.0);
  return short_ca_weight * raw_center_ca + (1.0 - short_ca_weight) * bearing_enhanced;
}

const Eigen::Vector3d & Target::observed_center() const { return observed_center_; }

std::chrono::steady_clock::time_point Target::observation_time() const
{
  return observation_time_;
}

// 返回所有装甲板的预测四维状态 (X, Y, Z, Angle) 列表
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  const Eigen::Vector3d center_fused = fused_center();

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (i == 1 || i == 3);

    double r = (use_l_h) ? ekf_.x[8] + ekf_.x[9] : ekf_.x[8];
    double plate_x = center_fused.x() - r * std::cos(angle);
    double plate_y = center_fused.y() - r * std::sin(angle);

    double plate_z;
    if (name == ArmorName::outpost) {
      double dz = tower_armor_hs[i].second - tower_armor_hs[0].second;
      int dz_px = dz > 0 ? 1 : -1;
      int dz_mu;
      if (std::abs(dz) > TOWER_ARMOR_DTB) {
        dz_mu = 2;
      } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
        dz_mu = 1;
      } else {
        dz_mu = 0;
      }
      plate_z = center_fused.z() + ekf_.x[10] * dz_px * dz_mu;
    } else {
      plate_z = (use_l_h) ? center_fused.z() + ekf_.x[10] : center_fused.z();
    }

    _armor_xyza_list.push_back({plate_x, plate_y, plate_z, angle});
  }
  return _armor_xyza_list;
}

// 检查滤波器半径是否发散
bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  if (armor_num_ != 4) return !r_ok;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
  if (r_ok && l_ok) return false;
  return true;
}

// 判断当前目标是否收敛
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

// 核心函数：根据 EKF 状态和 ID 推算该装甲板在世界坐标系下的理论位置 XYZ
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
      
      // 使用定义的常量区分大跳变和小跳变阶梯
      if (std::abs(dz) > TOWER_ARMOR_DTB) {
        dz_mu = 2; // 相隔两个阶梯 (大跳变)
      } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
        dz_mu = 1; // 相隔一个阶梯 (小跳变)
      } else {
        dz_mu = 0; // 同一阶梯
      }
      
      // 结合滤波器的高度参数 x[10]
      armor_z = x[4] + x[10] * dz_px * dz_mu; 
    } else {
      armor_z = (use_l_h) ? x[4] + x[10] : x[4];
    }
  return {armor_x, armor_y, armor_z};
}

// 计算当前预测观测函数的雅可比矩阵
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
    
    // 使用定义的常量区分跳变阶梯
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
  
  // 11 维位置偏导雅可比矩阵
  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
  H_armor_xyza(0, 0) = 1; H_armor_xyza(0, 6) = dx_da; H_armor_xyza(0, 8) = dx_dr; H_armor_xyza(0, 9) = dx_dl;
  H_armor_xyza(1, 2) = 1; H_armor_xyza(1, 6) = dy_da; H_armor_xyza(1, 8) = dy_dr; H_armor_xyza(1, 9) = dy_dl;
  H_armor_xyza(2, 4) = 1; H_armor_xyza(2, 10) = dz_dh;
  H_armor_xyza(3, 6) = 1;

  // 将 XYZ 偏导转换到 YPD 球坐标系下
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);

  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };

  // 链式求导法 H_Final = H_ypda * H_xyza
  return H_armor_ypda * H_armor_xyza;
}

// 检查是否完成初始化
bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
