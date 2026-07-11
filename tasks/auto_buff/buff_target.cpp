#include "buff_target.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/core.h>

namespace auto_buff
{
namespace
{
Eigen::Vector3d point_buff2world_from_state(
  const Eigen::VectorXd & x, const Eigen::Vector3d & point_in_buff)
{
  Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(x[4], 0.0, x[5]));  // pitch = 0
  Eigen::Vector3d center_in_world = tools::ypd2xyz(Eigen::Vector3d(x[0], x[2], x[3]));
  return R_buff2world * point_in_buff + center_in_world;
}

bool should_reset_track(
  const Eigen::VectorXd & x, const PowerRune & p, int last_track_id,
  double blade_error_gate_m, double roll_error_gate_rad, std::string & reason)
{
  if (p.track_id >= 0 && last_track_id >= 0 && p.track_id != last_track_id) {
    reason = fmt::format("track {}->{}", last_track_id, p.track_id);
    return true;
  }

  if (x.size() < 6) return false;

  const auto predicted_blade =
    point_buff2world_from_state(x, Eigen::Vector3d(0.0, 0.0, RUNE_RADIUS_M));
  const double blade_error = (predicted_blade - p.blade_xyz_in_world).norm();
  const double quality_scale = std::sqrt(std::max(1.0, p.measurement_noise_scale));
  if (blade_error > blade_error_gate_m * quality_scale) {
    reason = fmt::format("blade_err {:.3f}m", blade_error);
    return true;
  }

  const double roll_error = std::abs(tools::limit_rad(p.ypr_in_world[2] - x[5]));
  if (roll_error > roll_error_gate_rad * quality_scale) {
    reason = fmt::format("roll_err {:.1f}deg", roll_error * 57.3);
    return true;
  }

  return false;
}

int configured_small_buff_direction()
{
  if (SMALL_BUFF_DIRECTION > 0) return 1;
  if (SMALL_BUFF_DIRECTION < 0) return -1;
  return 0;
}

int small_buff_direction(int auto_direction)
{
  const int configured_direction = configured_small_buff_direction();
  if (configured_direction != 0) return configured_direction;
  if (auto_direction > 0) return 1;
  if (auto_direction < 0) return -1;
  return 0;
}

int small_roll_direction(int image_direction, int positive_roll_image_direction)
{
  const int direction = small_buff_direction(image_direction);
  if (direction == 0) return 0;
  const int roll_image_direction = positive_roll_image_direction == 0 ? 1 : positive_roll_image_direction;
  return direction * roll_image_direction;
}

int signum(int value)
{
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}

double unwrap_near(double angle, double reference)
{
  return angle + std::round((reference - angle) / CV_2PI) * CV_2PI;
}
}  // namespace

///voter

Voter::Voter() : clockwise_(0) {}

void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return;
  if (tools::limit_rad(angle_now - angle_last) < 0.0)
    clockwise_--;
  else
    clockwise_++;
}

int Voter::clockwise() { return clockwise_ > 0 ? 1 : -1; }

/// Target

Target::Target() : first_in_(true), unsolvable_(true) {};

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_) return Eigen::Vector3d(0, 0, 0);
  const Eigen::Vector3d center_in_world =
    tools::ypd2xyz(Eigen::Vector3d(ekf_.x[0], ekf_.x[2], ekf_.x[3]));
  return rotation_buff2world() * point_in_buff + center_in_world;
}

Eigen::Matrix3d Target::rotation_buff2world() const
{
  return tools::rotation_matrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5]));
}

bool Target::is_unsolve() const { return unsolvable_; }

bool Target::is_blind() const { return blind_; }

bool Target::can_fire(std::chrono::steady_clock::time_point now) const
{
  if (
    unsolvable_ || blind_ || !has_full_observation_timestamp_ ||
    last_pose_quality_ != BuffPoseQuality::FULL_8_POINT) {
    return false;
  }
  return tools::delta_time(now, last_full_observation_timestamp_) <=
         BUFF_FIRE_FULL_OBSERVATION_MAX_AGE_S;
}

double Target::relative_time(std::chrono::steady_clock::time_point timestamp)
{
  if (!has_start_timestamp_) {
    start_timestamp_ = timestamp;
    has_start_timestamp_ = true;
  }
  return tools::delta_time(timestamp, start_timestamp_);
}

bool Target::predict_without_measurement(std::chrono::steady_clock::time_point timestamp)
{
  if (
    first_in_ || !has_measurement_timestamp_ ||
    tools::delta_time(timestamp, last_measurement_timestamp_) > BUFF_BLIND_TIMEOUT_S) {
    unsolvable_ = true;
    blind_ = true;
    return false;
  }

  const double nowtime = relative_time(timestamp);
  predict(std::max(0.0, nowtime - lasttime_));
  lasttime_ = nowtime;
  unsolvable_ = false;
  blind_ = true;
  return true;
}

void Target::record_measurement(
  const PowerRune & p, std::chrono::steady_clock::time_point timestamp)
{
  has_measurement_timestamp_ = true;
  last_measurement_timestamp_ = timestamp;
  last_pose_quality_ = p.pose_quality;
  if (p.pose_quality == BuffPoseQuality::FULL_8_POINT) {
    has_full_observation_timestamp_ = true;
    last_full_observation_timestamp_ = timestamp;
  }
  blind_ = false;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

/// SmallTarget

SmallTarget::SmallTarget() : Target() {}

Eigen::Matrix3d SmallTarget::rotation_buff2world() const
{
  if (!has_plane_basis_) return Target::rotation_buff2world();

  Eigen::Vector3d z =
    std::cos(ekf_.x[5]) * phase_zero_axis_ + std::sin(ekf_.x[5]) * phase_quarter_axis_;
  z.normalize();
  Eigen::Vector3d x = plane_normal_.normalized();
  Eigen::Vector3d y = z.cross(x).normalized();
  x = y.cross(z).normalized();

  Eigen::Matrix3d rotation;
  rotation.col(0) = x;
  rotation.col(1) = y;
  rotation.col(2) = z;
  return rotation;
}

void SmallTarget::update_plane_basis(const PowerRune & p, bool initialize)
{
  Eigen::Vector3d normal = p.plane_normal_in_world;
  if (!normal.allFinite() || normal.norm() < 1e-6) {
    if (has_plane_basis_) return;
    normal = Eigen::Vector3d::UnitX();
  }
  normal.normalize();

  if (has_plane_basis_) {
    if (normal.dot(plane_normal_) < 0.0) normal = -normal;
  } else if (normal.dot(p.xyz_in_world) > 0.0) {
    // PnP平面法向存在正负二义性，固定为朝向相机的一侧。
    normal = -normal;
  }

  Eigen::Vector3d previous_radial = Eigen::Vector3d::Zero();
  double previous_phase = 0.0;
  if (has_plane_basis_ && ekf_.x.size() >= 6) {
    previous_phase = ekf_.x[5];
    previous_radial = std::cos(previous_phase) * phase_zero_axis_ +
                      std::sin(previous_phase) * phase_quarter_axis_;
  }

  if (!has_plane_basis_ || initialize || plane_normal_weight_ <= 0.0) {
    plane_normal_sum_ = normal;
    plane_normal_weight_ = 1.0;
  } else {
    const double weight = p.pose_quality == BuffPoseQuality::FULL_8_POINT ? 1.0 : 0.2;
    plane_normal_sum_ += weight * normal;
    plane_normal_weight_ += weight;
    normal = plane_normal_sum_.normalized();
  }

  Eigen::Vector3d zero_axis = Eigen::Vector3d::UnitZ() - normal.z() * normal;
  if (zero_axis.norm() < 0.1) {
    zero_axis = Eigen::Vector3d::UnitX() - normal.x() * normal;
  }
  zero_axis.normalize();
  Eigen::Vector3d quarter_axis = normal.cross(zero_axis).normalized();

  plane_normal_ = normal;
  phase_zero_axis_ = zero_axis;
  phase_quarter_axis_ = quarter_axis;
  has_plane_basis_ = true;

  if (previous_radial.norm() > 1e-6) {
    const double reexpressed = std::atan2(
      previous_radial.dot(phase_quarter_axis_), previous_radial.dot(phase_zero_axis_));
    ekf_.x[5] = unwrap_near(reexpressed, previous_phase);
  }
}

double SmallTarget::measure_phase(const PowerRune & p, double reference) const
{
  Eigen::Vector3d radial = p.blade_xyz_in_world - p.xyz_in_world;
  if (!radial.allFinite() || radial.norm() < 1e-6 || !has_plane_basis_) return reference;
  radial.normalize();
  const double wrapped =
    std::atan2(radial.dot(phase_quarter_axis_), radial.dot(phase_zero_axis_));
  return unwrap_near(wrapped, reference);
}

// void SmallTarget::get_target(
//   const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
// {
//   static int lost_cn = 0;
//   static std::chrono::steady_clock::time_point start_timestamp = timestamp;
//   auto time_gap = tools::delta_time(timestamp, start_timestamp);

//   // 1. 处理视觉丢帧（触发盲推或彻底丢失）
//   if (!p.has_value()) {
//     lost_cn++;
//     // 如果丢失超过容忍阈值(6帧)或系统尚未初始化，放弃解算
//     if (lost_cn > 5 || first_in_) {
//       unsolvable_ = true;
//       first_in_ = true;
//       // tools::logger()->debug("[Target] 小符丢失过久，停止盲推");
//     } else {
//       // 短暂丢失，触发盲推机制
//       predict(time_gap - lasttime_); // 纯推演卡尔曼先验状态，不引入测量更新
//       lasttime_ = time_gap;          // 更新时间戳，供下次计算 dt 使用
//       unsolvable_ = false;           // 保持可解算状态，让 Aimer 继续下发指令
//       tools::logger()->debug("[Target] 小符盲推中，连续丢失帧数: {}", lost_cn);
//     }
//     return; // 盲推结束，直接返回
//   }

//   // 2. 正常识别到目标，重置丢失计数
//   lost_cn = 0;

//   // 3. 冷启动初始化
//   if (first_in_) {
//     unsolvable_ = true;
//     init(time_gap, p.value());
//     first_in_ = false;
//   }

//   // 4. 正常卡尔曼滤波测量更新
//   unsolvable_ = false;
//   update(time_gap, p.value());

//   // 5. 处理滤波器发散
//   if (std::abs(ekf_.x[6]) > SMALL_W + CV_PI / 18 || std::abs(ekf_.x[6]) < SMALL_W - CV_PI / 18) {
//     unsolvable_ = true;
//     tools::logger()->debug("[Target] 小符角度发散spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
//     first_in_ = true;
//     return;
//   }
// }

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  const double time_gap = relative_time(timestamp);

  if (!p.has_value()) {
    if (!has_stable_small_prediction_direction() || !predict_without_measurement(timestamp)) {
      unsolvable_ = true;
    }
    return;
  }

  const auto & obs = p.value();

  // init
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, obs);
    first_in_ = false;
    last_track_id_ = obs.track_id;
    record_measurement(obs, timestamp);
  }

  std::string reset_reason;
  bool reset_track = false;
  if (obs.track_id >= 0 && last_track_id_ >= 0 && obs.track_id != last_track_id_) {
    if (obs.pose_quality != BuffPoseQuality::FULL_8_POINT) {
      tools::logger()->debug("[Target] 小符拒绝部分观测换轨");
      predict_without_measurement(timestamp);
      return;
    }

    const int old_track_id = last_track_id_;
    update_plane_basis(obs, false);
    const double switched_phase = measure_phase(obs, ekf_.x[5]);
    ekf_.x[5] = switched_phase;
    ekf_.P.row(5).setZero();
    ekf_.P.col(5).setZero();
    ekf_.P(5, 5) = 0.01;
    has_last_observed_phase_ = true;
    last_observed_phase_ = switched_phase;
    lasttime_ = time_gap;
    last_track_id_ = obs.track_id;
    unsolvable_ = !has_stable_small_prediction_direction();
    record_measurement(obs, timestamp);
    tools::logger()->debug(
      "[Target] 小符切换跟踪 {}->{}, 保留中心状态", old_track_id, obs.track_id);
    return;
  } else {
    const double observed_phase = measure_phase(obs, ekf_.x[5]);
    const double quality_scale = std::sqrt(std::max(1.0, obs.measurement_noise_scale));
    const double phase_error = std::abs(observed_phase - ekf_.x[5]);
    if (phase_error > CV_PI / 9.0 * quality_scale) {
      reset_reason = fmt::format("phase_err {:.1f}deg", phase_error * 57.3);
      reset_track = true;
    }
  }
  if (reset_track) {
    tools::logger()->debug("[Target] 小符拒绝异常观测: {}", reset_reason);
    predict_without_measurement(timestamp);
    return;
  }
  innovation_reject_count_ = 0;

  // kalman update
  unsolvable_ = false;
  update(time_gap, obs);
  last_track_id_ = obs.track_id;
  record_measurement(obs, timestamp);

  // 处理发散
  if (
    std::abs(ekf_.x[6]) > 1e-6 &&
    (std::abs(ekf_.x[6]) > SMALL_W + CV_PI / 18 ||
     std::abs(ekf_.x[6]) < SMALL_W - CV_PI / 18)) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 小符角度发散spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
    first_in_ = true;
    return;
  }

  if (!has_stable_small_prediction_direction()) {
    unsolvable_ = true;
  }
}

void SmallTarget::predict(double dt)
{
  dt = std::max(0.0, dt);
  ekf_.x[6] = small_prediction_roll_direction() * SMALL_W;

  // R中心在世界系中近似静止。保留一个强阻尼yaw速度，只用于吸收短时平台平移和外参误差。
  const double center_velocity_decay = std::exp(-dt / 0.10);
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, // R_yaw
        0.0, center_velocity_decay, 0.0, 0.0, 0.0, 0.0, 0.0, // R_v_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, // roll
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0; // spd

  // 小符角速度固定；中心、平面朝向只允许缓慢变化。
  const double v1 = 0.0002;
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Q_ << a * v1 + 1e-6, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 1e-6, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 1e-3, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 1e-7, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 1e-6, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  // clang-format on 
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    return x_prior;
  };
  ekf_.predict(A_, Q_, f);
  ekf_.x[6] = small_prediction_roll_direction() * SMALL_W;
}

void SmallTarget::init(double nowtime, const PowerRune & p)
{
  // 初始化内部变量
  lasttime_ = nowtime;

  // 初始状态协方差矩阵
  x0_.resize(7);
  P0_.resize(7, 7);
  A_.resize(7, 7);
  Q_.resize(7, 7);
  H_.resize(7, 7);//z x
  R_.resize(7, 7);//z z
  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row]
  // [spd]   w=CV_PI/6

  update_plane_basis(p, !has_plane_basis_);
  const double phase_reference = ekf_.x.size() >= 6 ? ekf_.x[5] : 0.0;
  const double observed_phase = measure_phase(p, phase_reference);
  const double plane_yaw = std::atan2(plane_normal_.y(), plane_normal_.x());

  // clang-format off
  // 初始状态
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         plane_yaw, observed_phase,
         SMALL_W * small_prediction_roll_direction();
  // 初始状态协方差矩阵
  P0_ << 0.01,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 0.01,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 0.01,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 0.25,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 0.04,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 0.04,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 1e-4;
  // 状态转移矩阵
  // A_ 
  // 过程噪声协方差矩阵                            //// 调整
  // Q_ 
  // 测量方程矩阵
  // H_
  // 测量噪声协方差矩阵                            //// 调整
  // R_

  // clang-format on

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    return c;
  };
  // 创建扩展卡尔曼滤波器对象
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
  has_last_observed_phase_ = true;
  last_observed_phase_ = observed_phase;
}

void SmallTarget::update(double nowtime, const PowerRune & p)
{
  // [R_yaw]     angle0
  // [v_R_yaw]
  // [R_pitch]   angle2
  // [R_dis]
  // [yaw]       angle4
  // [angle/row] angle5
  // [spd]   w=CV_PI/6
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;  // R
  update_plane_basis(p, false);
  double observed_phase = measure_phase(p, ekf_.x[5]);
  update_observed_small_direction(observed_phase);

  ekf_.x[6] = SMALL_W * small_prediction_roll_direction();

  // 预测下一个状态
  predict(nowtime - lasttime_);
  observed_phase = unwrap_near(observed_phase, ekf_.x[5]);
  const double plane_yaw = std::atan2(plane_normal_.y(), plane_normal_.x());

  // clang-format off
  Eigen::MatrixXd H{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_yaw
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // R_pitch
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, // R_dis
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, // plane_yaw
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}  // continuous phase
  };

  Eigen::MatrixXd R{
    {0.0001,    0.0,  0.0, 0.0,    0.0}, // R_yaw
    {   0.0, 0.0001,  0.0, 0.0,    0.0}, // R_pitch
    {   0.0,    0.0, 0.04, 0.0,    0.0}, // R_dis
    {   0.0,    0.0,  0.0, 0.01,   0.0}, // plane_yaw
    {   0.0,    0.0,  0.0, 0.0, 0.0001}  // continuous phase
  };
  R *= p.measurement_noise_scale;
  // clang-format on

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z{{R_ypd[0], R_ypd[1], R_ypd[2], plane_yaw, observed_phase}};
  ekf_.update(z, H, R, z_subtract);

  ekf_.x[6] = SMALL_W * small_prediction_roll_direction();

  // 更新lasttime
  lasttime_ = nowtime;
  return;
}

void SmallTarget::update_observed_small_direction(double observed_phase)
{
  if (!has_last_observed_phase_) {
    has_last_observed_phase_ = true;
    last_observed_phase_ = observed_phase;
    return;
  }

  const double observed_delta = observed_phase - last_observed_phase_;
  last_observed_phase_ = observed_phase;

  if (configured_small_buff_direction() != 0) return;

  constexpr double min_direction_delta = CV_PI / 900.0;  // 0.2 deg, reject jitter
  constexpr double max_direction_delta = CV_PI / 5.0;    // reject slot switches
  constexpr double confirm_window_delta = CV_PI / 60.0;  // 3 deg net motion
  constexpr int direction_window = 10;
  constexpr int min_window_samples = 6;
  constexpr int confirm_vote_margin = 4;
  constexpr int confirm_score = 6;
  constexpr int score_limit = 30;
  constexpr int reverse_confirm_frames = 16;

  if (
    std::abs(observed_delta) <= min_direction_delta ||
    std::abs(observed_delta) >= max_direction_delta) {
    return;
  }

  const int phase_direction = observed_delta > 0.0 ? 1 : -1;
  small_direction_deltas_.push_back(observed_delta);
  small_direction_votes_.push_back(phase_direction);
  while (static_cast<int>(small_direction_deltas_.size()) > direction_window) {
    small_direction_deltas_.pop_front();
    small_direction_votes_.pop_front();
  }

  small_direction_score_ =
    std::clamp(small_direction_score_ + phase_direction, -score_limit, score_limit);

  double window_delta_sum = 0.0;
  int window_vote_sum = 0;
  for (size_t i = 0; i < small_direction_deltas_.size(); ++i) {
    window_delta_sum += small_direction_deltas_[i];
    window_vote_sum += small_direction_votes_[i];
  }

  int window_direction = 0;
  if (
    static_cast<int>(small_direction_deltas_.size()) >= min_window_samples &&
    std::abs(window_delta_sum) >= confirm_window_delta &&
    std::abs(window_vote_sum) >= confirm_vote_margin) {
    const int delta_direction = window_delta_sum > 0.0 ? 1 : -1;
    const int vote_direction = signum(window_vote_sum);
    if (delta_direction == vote_direction) window_direction = delta_direction;
  }

  if (window_direction == 0) {
    small_reverse_candidate_direction_ = 0;
    small_reverse_confirm_count_ = 0;
    return;
  }

  if (small_auto_direction_ == 0) {
    const int score_direction = signum(small_direction_score_);
    if (
      score_direction == window_direction &&
      std::abs(small_direction_score_) >= confirm_score) {
      small_auto_direction_ = window_direction;
      small_direction_score_ =
        std::clamp(small_direction_score_, -score_limit, score_limit);
      small_reverse_candidate_direction_ = 0;
      small_reverse_confirm_count_ = 0;
      tools::logger()->debug(
        "[Target] 小符连续相位方向确认: dir {}, score {}, window_votes {}, window_delta {:.1f}deg",
        small_auto_direction_, small_direction_score_, window_vote_sum,
        window_delta_sum * 57.3);
    }
    return;
  }

  if (window_direction == small_auto_direction_) {
    small_reverse_candidate_direction_ = 0;
    small_reverse_confirm_count_ = 0;
    return;
  }

  if (small_reverse_candidate_direction_ != window_direction) {
    small_reverse_candidate_direction_ = window_direction;
    small_reverse_confirm_count_ = 1;
  } else {
    small_reverse_confirm_count_++;
  }

  if (
    small_reverse_confirm_count_ >= reverse_confirm_frames &&
    small_direction_score_ * small_auto_direction_ <= -confirm_score) {
    small_auto_direction_ = window_direction;
    small_direction_score_ = window_direction * confirm_score;
    small_reverse_candidate_direction_ = 0;
    small_reverse_confirm_count_ = 0;
    tools::logger()->debug(
      "[Target] 小符连续相位方向重确认: dir {}, score {}, window_votes {}, window_delta {:.1f}deg",
      small_auto_direction_, small_direction_score_, window_vote_sum, window_delta_sum * 57.3);
  }
}

int SmallTarget::small_prediction_roll_direction() const
{
  return small_buff_direction(small_auto_direction_);
}

bool SmallTarget::has_stable_small_prediction_direction() const
{
  return small_buff_direction(small_auto_direction_) != 0;
}

/// BigTarget

BigTarget::BigTarget() : Target(), spd_fitter_(100, 0.5, 1.884, 2.000) {}

// void BigTarget::get_target(
//   const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
// {
//   static int lost_cn = 0;
//   static std::chrono::steady_clock::time_point start_timestamp = timestamp;
//   auto time_gap = tools::delta_time(timestamp, start_timestamp);

//   // 1. 处理视觉丢帧（触发盲推或彻底丢失）
//   if (!p.has_value()) {
//     lost_cn++;
//     // 如果丢失超过容忍阈值(6帧)或系统尚未初始化，放弃解算
//     if (lost_cn > 60 || first_in_) {
//       unsolvable_ = true;
//       first_in_ = true;
//       // tools::logger()->debug("[Target] 大符丢失过久，停止盲推");
//     } else {
//       // 短暂丢失，触发盲推机制
//       predict(time_gap - lasttime_); // 纯推演卡尔曼先验状态
//       lasttime_ = time_gap;          // 更新时间戳
//       unsolvable_ = false;           // 保持可解算状态
//       tools::logger()->debug("[Target] 大符盲推中，连续丢失帧数: {}", lost_cn);
//     }
//     return; // 盲推结束，直接返回
//   }

//   // 2. 正常识别到目标，重置丢失计数
//   lost_cn = 0;

//   // 3. 冷启动初始化
//   if (first_in_) {
//     unsolvable_ = true;
//     init(time_gap, p.value());
//     first_in_ = false;
//   }

//   // 4. 正常卡尔曼滤波测量更新
//   unsolvable_ = false;
//   update(time_gap, p.value());

//   // 5. 处理滤波器发散
//   if (
//     ekf_.x[7] > 1.045 * 1.5 || ekf_.x[7] < 0.78 / 1.5 || ekf_.x[8] > 2.0 * 1.5 ||
//     ekf_.x[8] < 1.884 / 1.5) {
//     unsolvable_ = true;
//     tools::logger()->debug("[Target] 大符角度发散a: {:.2f}b:{:.2f}", ekf_.x[7], ekf_.x[8]);
//     first_in_ = true;
//     return;
//   }
// }

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  const double time_gap = relative_time(timestamp);
  if (!p.has_value()) {
    predict_without_measurement(timestamp);
    return;
  }

  const auto & obs = p.value();

  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, obs);
    first_in_ = false;
    last_track_id_ = obs.track_id;
    speed_model_track_id_ = obs.track_id;
    record_measurement(obs, timestamp);
  }

  std::string reset_reason;
  if (should_reset_track(ekf_.x, obs, last_track_id_, 0.50, CV_PI / 6.0, reset_reason)) {
    const bool track_changed =
      obs.track_id >= 0 && last_track_id_ >= 0 && obs.track_id != last_track_id_;
    if (obs.pose_quality != BuffPoseQuality::FULL_8_POINT ||
        (!track_changed && ++innovation_reject_count_ < 3)) {
      tools::logger()->debug("[Target] 大符拒绝异常观测: {}", reset_reason);
      predict_without_measurement(timestamp);
      return;
    }
    tools::logger()->debug("[Target] 大符重置EKF: {}", reset_reason);
    reset_count_++;
    spd_fitter_.clear();
    has_last_speed_observation_ = false;
    init(time_gap, obs);
    first_in_ = false;
    unsolvable_ = false;
    last_track_id_ = obs.track_id;
    speed_model_track_id_ = obs.track_id;
    innovation_reject_count_ = 0;
    record_measurement(obs, timestamp);
    return;
  }
  innovation_reject_count_ = 0;

  unsolvable_ = false;
  update(time_gap, obs);
  last_track_id_ = obs.track_id;
  record_measurement(obs, timestamp);
}

void BigTarget::predict(double dt)
{
  dt = std::max(0.0, dt);
  const bool fit_ready = spd_fitter_.ready(60, 0.5, 0.70);
  const auto fit = spd_fitter_.best_result_;
  const double a = fit_ready ? fit.A : ekf_.x[7];
  const double w = fit_ready ? fit.omega : ekf_.x[8];
  const double fi = fit_ready ? fit.phi : ekf_.x[9];
  const double c = fit_ready ? fit.C : 2.09 - a;
  double t = lasttime_ + dt;
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_yaw
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//v_R_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,//yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, voter.clockwise() * dt , 0.0, 0.0, 0.0,//row
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sin(w * t + fi) - 1, t * a * cos(w * t + fi), a * cos(w * t + fi),//spd
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,//a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,//w
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;//theta
        
  // 过程噪声协方差矩阵                            //// 调整
  auto v1 = 0.9;  // 角加速度方差
  auto a1 = dt * dt * dt * dt / 4;
  auto b1 = dt * dt * dt / 2;
  auto c1 = dt * dt;
  Q_ << a1 * v1, b1 * v1, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        b1 * v1, c1 * v1, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0, 0.09,  0.0,  0.0,  0.0,  0.0,//row
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.5,  0.0,  0.0,  0.0,// spd 0.5  1
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,// a
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,// w
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  1.0;// fi
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,// spd  2
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  4.0;

            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0;
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = x;
    x_prior[0] = tools::limit_rad(x_prior[0] + dt * x_prior[1]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]); // yaw
    x_prior[5] = tools::limit_rad(x_prior[5] + voter.clockwise() * 
    (-a / w * std::cos(w * t + fi) + a / w * std::cos(w * lasttime_ + fi) + c * dt)); // roll
    x_prior[6] = std::clamp(a * sin(w * t + fi) + c, 0.0, 2.1); // spd
    if (fit_ready) {
      x_prior[7] = a;
      x_prior[8] = w;
      x_prior[9] = tools::limit_rad(fi);
    }
    return x_prior;
  };
  // clang-format on
  ekf_.predict(A_, Q_, f);
  lasttime_ = t;
}

void BigTarget::init(double nowtime, const PowerRune & p)
{
  // 初始化内部变量
  lasttime_ = nowtime;
  unsolvable_ = true;
  fit_spd_ = 1.1775;
  has_last_speed_observation_ = false;

  // 初始状态协方差矩阵
  x0_.resize(10);
  P0_.resize(10, 10);
  A_.resize(10, 10);
  Q_.resize(10, 10);
  H_.resize(7, 10);
  R_.resize(7, 7);

  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row]
  // [spd]       角速度 a*sin(wt) + 2.09 - a
  // [a]         0.78-1.045
  // [w]         1.884-2.000
  // [fi]

  // clang-format off
  // 初始状态
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], p.ypr_in_world[2], 
         1.1775, 0.9125, 1.942, 0.0;//std::atan((spd - 2.09) / 0.9125 + 1
  // 初始状态协方差矩阵
  P0_ << 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 100.0, 0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 400.0;
  // 状态转移矩阵
  // A_
  // 过程噪声协方差矩阵                            //// 调整
  // Q_
  // 测量方程矩阵
  // H_
  // 测量噪声协方差矩阵                            //// 调整
  // R_

  // clang-format on

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    c[9] = tools::limit_rad(c[9]);
    return c;
  };
  // 创建扩展卡尔曼滤波器对象
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void BigTarget::update(double nowtime, const PowerRune & p)
{
  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row] 角度
  // [spd]       角速度 a*sin(wt) + 2.09 - a
  // [a]         0.78-1.045
  // [w]         1.884-2.000
  // [fi]
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;  // R
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;  // center of blade

  if (speed_model_track_id_ != p.track_id) {
    spd_fitter_.clear();
    has_last_speed_observation_ = false;
    speed_model_track_id_ = p.track_id;
  }

  if (has_last_speed_observation_) {
    const double observation_dt = nowtime - last_speed_observation_time_;
    if (observation_dt > 1e-4 && observation_dt < 0.1) {
      const double signed_speed =
        tools::limit_rad(ypr[2] - last_speed_observation_roll_) / observation_dt;
      const double speed = std::abs(signed_speed);
      if (speed <= 2.1) {
        fit_spd_ = 0.85 * fit_spd_ + 0.15 * speed;
        spd_fitter_.add_data(nowtime, speed);
        spd_fitter_.fit();
      }
    }
  }
  voter.vote(ekf_.x[5], ypr[2]);
  has_last_speed_observation_ = true;
  last_speed_observation_roll_ = ypr[2];
  last_speed_observation_time_ = nowtime;

  const auto anglelast = ekf_.x[5];

  // 预测下一个状态
  const double state_dt = nowtime - lasttime_;
  predict(state_dt);

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3
  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  /// 1.

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3

  // clang-format off
  Eigen::MatrixXd H1{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_yaw
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_pitch
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_dis
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}  // roll
  };

  Eigen::MatrixXd R1{
    {0.01, 0.0, 0.0,  0.0}, // R_yaw
    {0.0, 0.01, 0.0,  0.0}, // R_pitch
    {0.0,  0.0, 0.5,  0.0}, // R_dis
    {0.0,  0.0, 0.0, 0.02}  // roll  1: 0.01 2:0.04
  };
  R1 *= p.measurement_noise_scale;
  // clang-format on

  // 防止夹角求差出现异常值
  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;  //4 1
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], ypr[2]}};  // R_ypd roll

  ekf_.update(z1, H1, R1, z_subtract1);

  ///2.

  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  // clang-format off
  Eigen::MatrixXd H2 = h_jacobian();  // 3*10

  Eigen::MatrixXd R2{
    {0.01, 0.0, 0.0}, // B_yaw
    {0.0, 0.01, 0.0}, // B_pitch
    {0.0,  0.0, 0.5}  // B_dis
  };
  R2 *= p.measurement_noise_scale;
  // clang-format on

  // 定义非线性转换函数h: x -> z
  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    Eigen::Vector3d B_xyz =
      point_buff2world_from_state(x, Eigen::Vector3d(0.0, 0.0, RUNE_RADIUS_M));
    return tools::xyz2ypd(B_xyz);
  };

  // 防止夹角求差出现异常值
  auto z_subtract2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;  //6 1
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    return c;
  };

  Eigen::VectorXd z2{{B_ypd[0], B_ypd[1], B_ypd[2]}};

  ekf_.update(z2, H2, R2, h2, z_subtract2);

  if (spd_fitter_.ready(60, 0.5, 0.70)) {
    const auto & fit = spd_fitter_.best_result_;
    fit_spd_ = std::clamp(
      spd_fitter_.sine_function(nowtime, fit.A, fit.omega, fit.phi, fit.C), 0.0, 2.1);
  }
  spd = state_dt > 1e-4
          ? voter.clockwise() * tools::limit_rad(ekf_.x[5] - anglelast) / state_dt
          : 0.0;

  // 更新lasttime
  lasttime_ = nowtime;
  unsolvable_ = false;
  return;
}

Eigen::MatrixXd BigTarget::h_jacobian() const
{
  /// Z(3,1) = H3(3,3) * H2(3,5) * H1(5,5) * H0(5,10) * x(10,1)

  // clang-format off
  Eigen::MatrixXd H0{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}
  };// 5*7

  Eigen::VectorXd R_ypd{{ekf_.x[0], ekf_.x[2], ekf_.x[3]}};
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);  // 3*3
  Eigen::MatrixXd H1{
    {H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0},
    {H_ypd2xyz(1, 0), H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0},
    {H_ypd2xyz(2, 0), H_ypd2xyz(2, 1), H_ypd2xyz(2, 2), 0.0, 0.0},
    {            0.0,             0.0,             0.0, 1.0, 0.0},
    {            0.0,             0.0,             0.0, 0.0, 1.0}
  };// 5*5

  // double pitch = 0;
  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);
  Eigen::MatrixXd H2{
    {1.0, 0.0, 0.0, RUNE_RADIUS_M * cos_yaw * sin_roll,  RUNE_RADIUS_M * sin_yaw * cos_roll},
    {0.0, 1.0, 0.0, RUNE_RADIUS_M * sin_yaw * sin_roll, -RUNE_RADIUS_M * cos_yaw * cos_roll},
    {0.0, 0.0, 1.0,                                  0.0,               -RUNE_RADIUS_M * sin_roll}
  };// 3*5

  Eigen::VectorXd B_xyz =
    point_buff2world_from_state(ekf_.x, Eigen::Vector3d(0.0, 0.0, RUNE_RADIUS_M));
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);// 3*3
  // clang-format on

  return H3 * H2 * H1 * H0;  // 3*7

  // auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
  //   Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
  //   Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
  //   Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
  //   Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  //   Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
  //   return B_ypd;
  // };
}
}  // namespace auto_buff
