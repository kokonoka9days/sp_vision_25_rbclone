#include "buff_target.hpp"

#include <algorithm>
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
  const Eigen::VectorXd & x, const PowerRune & p, int last_target_slot_id,
  double blade_error_gate_m, double roll_error_gate_rad, std::string & reason)
{
  if (p.target_slot_id >= 0 && last_target_slot_id >= 0 && p.target_slot_id != last_target_slot_id) {
    reason = fmt::format("slot {}->{}", last_target_slot_id, p.target_slot_id);
    return true;
  }

  if (x.size() < 6) return false;

  const auto predicted_blade =
    point_buff2world_from_state(x, Eigen::Vector3d(0.0, 0.0, RUNE_RADIUS_M));
  const double blade_error = (predicted_blade - p.blade_xyz_in_world).norm();
  if (blade_error > blade_error_gate_m) {
    reason = fmt::format("blade_err {:.3f}m", blade_error);
    return true;
  }

  const double roll_error = std::abs(tools::limit_rad(p.ypr_in_world[2] - x[5]));
  if (roll_error > roll_error_gate_rad) {
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
}  // namespace

///voter

Voter::Voter() : clockwise_(0) {}

void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return;
  if (angle_last > angle_now)
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
  return point_buff2world_from_state(ekf_.x, point_in_buff);
}

bool Target::is_unsolve() const { return unsolvable_; }

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

/// SmallTarget

SmallTarget::SmallTarget() : Target() {}

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
  static int lost_cn = 0;
  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  if (!p.has_value()) {
    lost_cn++;
    if (first_in_ || lost_cn > 6 || !has_stable_small_prediction_direction()) {
      if (!first_in_ && lost_cn == 7) tools::logger()->debug("[Target] 小符丢失buff");
      if (lost_cn > 6) first_in_ = true;
      unsolvable_ = true;
      return;
    }

    predict(std::max(0.0, time_gap - lasttime_));
    lasttime_ = time_gap;
    unsolvable_ = false;
    return;
  }

  const auto & obs = p.value();

  // init
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, obs);
    first_in_ = false;
    last_target_slot_id_ = obs.target_slot_id;
  }

  lost_cn = 0;

  std::string reset_reason;
  if (should_reset_track(ekf_.x, obs, last_target_slot_id_, 0.80, CV_PI / 9.0, reset_reason)) {
    tools::logger()->debug("[Target] 小符重置EKF: {}", reset_reason);
    update_positive_roll_image_direction(obs.positive_roll_image_direction);
    init(time_gap, obs);
    first_in_ = false;
    unsolvable_ = !has_stable_small_prediction_direction();
    lost_cn = 0;
    last_target_slot_id_ = obs.target_slot_id;
    return;
  }

  // kalman update
  unsolvable_ = false;
  update(time_gap, obs);
  last_target_slot_id_ = obs.target_slot_id;

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
  ekf_.x[6] = small_prediction_roll_direction() * SMALL_W;

  // 预测下一个状态
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, // R_yaw
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // R_v_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, // roll
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0; // spd

  // 过程噪声协方差矩阵。小符角速度是固定值，不让 spd 随滤波漂移。
  auto v1 = 0.001;  // R中心yaw角加速度方差
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Q_ << a * v1, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  // clang-format on 
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5]);
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

  // clang-format off
  // 初始状态
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], p.ypr_in_world[2], 
         SMALL_W * small_prediction_roll_direction();
  // 初始状态协方差矩阵
  P0_ << 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-2;
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
    return c;
  };
  // 创建扩展卡尔曼滤波器对象
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
  has_last_observed_target_angle_ = true;
  last_observed_target_angle_ = p.target_angle;
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
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;  // center of blade

  update_positive_roll_image_direction(p.positive_roll_image_direction);
  update_observed_small_direction(p);

  ekf_.x[6] = SMALL_W * small_prediction_roll_direction();

  // 预测下一个状态
  predict(nowtime - lasttime_);

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
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_yaw
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // R_pitch
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, // R_dis
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}  // roll
  };

  Eigen::MatrixXd R1{
    {0.01, 0.0, 0.0,  0.0}, // R_yaw
    {0.0, 0.01, 0.0,  0.0}, // R_pitch
    {0.0,  0.0, 0.5,  0.0}, // R_dis
    {0.0,  0.0, 0.0, 0.01}  // roll
  };
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
  Eigen::MatrixXd H2 = h_jacobian();  // 3*7

  Eigen::MatrixXd R2{
    {0.01, 0.0, 0.0}, // B_yaw
    {0.0, 0.01, 0.0}, // B_pitch
    {0.0,  0.0, 0.5}  // B_dis
  };
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

  ekf_.x[0] = R_ypd[0];
  ekf_.x[1] = 0.0;
  ekf_.x[2] = R_ypd[1];
  ekf_.x[3] = R_ypd[2];
  ekf_.x[4] = ypr[0];
  ekf_.x[5] = ypr[2];
  ekf_.x[6] = SMALL_W * small_prediction_roll_direction();

  // 更新lasttime
  lasttime_ = nowtime;
  return;
}

void SmallTarget::update_positive_roll_image_direction(int image_direction)
{
  if (image_direction == 0) return;
  last_positive_roll_image_direction_ = image_direction;
}

void SmallTarget::update_observed_small_direction(const PowerRune & p)
{
  if (!has_last_observed_target_angle_) {
    has_last_observed_target_angle_ = true;
    last_observed_target_angle_ = p.target_angle;
    return;
  }

  const double observed_delta = tools::limit_rad(p.target_angle - last_observed_target_angle_);
  has_last_observed_target_angle_ = true;
  last_observed_target_angle_ = p.target_angle;

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

  const int image_direction = observed_delta > 0.0 ? 1 : -1;
  small_direction_deltas_.push_back(observed_delta);
  small_direction_votes_.push_back(image_direction);
  while (static_cast<int>(small_direction_deltas_.size()) > direction_window) {
    small_direction_deltas_.pop_front();
    small_direction_votes_.pop_front();
  }

  small_direction_score_ =
    std::clamp(small_direction_score_ + image_direction, -score_limit, score_limit);

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
        "[Target] 小符图像方向确认: dir {}, score {}, window_votes {}, window_delta {:.1f}deg, roll_img_dir {}",
        small_auto_direction_, small_direction_score_, window_vote_sum,
        window_delta_sum * 57.3, last_positive_roll_image_direction_);
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
      "[Target] 小符图像方向重确认: dir {}, score {}, window_votes {}, window_delta {:.1f}deg, roll_img_dir {}",
      small_auto_direction_, small_direction_score_, window_vote_sum, window_delta_sum * 57.3,
      last_positive_roll_image_direction_);
  }
}

int SmallTarget::small_prediction_roll_direction() const
{
  const int image_direction = small_buff_direction(small_auto_direction_);
  if (image_direction == 0) return 0;
  return image_direction * last_positive_roll_image_direction_;
}

bool SmallTarget::has_stable_small_prediction_direction() const
{
  return small_buff_direction(small_auto_direction_) != 0;
}

Eigen::MatrixXd SmallTarget::h_jacobian() const
{
  /// Z(3,1) = H3(3,3) * H2(3,5) * H1(5,5) * H0(5,7) * x(7,1)

  // clang-format off
  Eigen::MatrixXd H0{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
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
  // 如果没有识别，退出函数
  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);
  const auto & obs = p.value();

  // init
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, obs);
    first_in_ = false;
    last_target_slot_id_ = obs.target_slot_id;
  }

  // 处理识别时间间隔过大
  if (lost_cn > 6) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 丢失buff");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  std::string reset_reason;
  if (should_reset_track(ekf_.x, obs, last_target_slot_id_, 0.25, CV_PI / 8.0, reset_reason)) {
    tools::logger()->debug("[Target] 大符重置EKF: {}", reset_reason);
    init(time_gap, obs);
    first_in_ = false;
    unsolvable_ = false;
    lost_cn = 0;
    last_target_slot_id_ = obs.target_slot_id;
    return;
  }

  // kalman update
  unsolvable_ = false;
  update(time_gap, obs);
  last_target_slot_id_ = obs.target_slot_id;

  // 处理发散
  if (
    ekf_.x[7] > 1.045 * 1.5 || ekf_.x[7] < 0.78 / 1.5 || ekf_.x[8] > 2.0 * 1.5 ||
    ekf_.x[8] < 1.884 / 1.5) {
    tools::logger()->debug("[Target] 大符角度发散a: {:.2f}b:{:.2f}", ekf_.x[7], ekf_.x[8]);
    first_in_ = true;
    return;
  }
}

void BigTarget::predict(double dt)
{
  // 预测下一个状态
  double spd = fit_spd_;
  // double spd = ekf_.x[6];
  double a = ekf_.x[7];
  double w = ekf_.x[8];
  double fi = ekf_.x[9];
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
    (-a / w * std::cos(w * t + fi) + a / w * std::cos(w * lasttime_ + fi) + (2.09 - a) * dt)); // roll
    x_prior[6] = a * sin(w * t + fi) + 2.09 - a; // spd
    return x_prior;
  };
  // clang-format on
  ekf_.predict(A_, Q_, f);
}

void BigTarget::init(double nowtime, const PowerRune & p)
{
  // 初始化内部变量
  lasttime_ = nowtime;
  unsolvable_ = true;

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

  bool is_jumped = false; // 新增：标记当前帧是否发生了目标跳变

  // 处理扇叶跳变 angle/row
  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    is_jumped = true; // 记录发生跳变
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        break;
      }
    }
  }

  // 修改：只有在连续跟踪同一个扇叶时，才信任观测差值并进行投票
  if (!is_jumped) {
    voter.vote(ekf_.x[5], ypr[2]);
  }

  auto anglelast = ekf_.x[5];  ///

  // 预测下一个状态
  predict(nowtime - lasttime_);

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

  // 对ekf速度进行最小二乘拟合 ekf_.x[6] -> fitting_speed -> predict position
  if (ekf_.x[6] < 2.1 && ekf_.x[6] >= 0) spd_fitter_.add_data(nowtime, ekf_.x[6]);
  spd_fitter_.fit();

  fit_spd_ = spd_fitter_.sine_function(
    nowtime, spd_fitter_.best_result_.A, spd_fitter_.best_result_.omega,
    spd_fitter_.best_result_.phi, spd_fitter_.best_result_.C);

  spd = voter.clockwise() * (ekf_.x[5] - anglelast) / (nowtime - lasttime_);  // 仅供调试
  spd = fit_spd_;
  if (std::abs(spd) > 4) spd = 0;

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
