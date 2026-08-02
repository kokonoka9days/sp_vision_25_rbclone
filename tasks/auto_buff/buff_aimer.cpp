#include "buff_aimer.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{
Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;      // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;  // degree to rad
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();
  if (yaml["buff_max_yaw_vel"]) max_yaw_vel_ = yaml["buff_max_yaw_vel"].as<double>();
  if (yaml["buff_max_pitch_vel"]) max_pitch_vel_ = yaml["buff_max_pitch_vel"].as<double>();
  if (yaml["buff_max_yaw_acc"]) max_yaw_acc_ = yaml["buff_max_yaw_acc"].as<double>();
  if (yaml["buff_max_pitch_acc"]) max_pitch_acc_ = yaml["buff_max_pitch_acc"].as<double>();
  if (yaml["buff_command_position_gain"]) {
    command_position_gain_ = yaml["buff_command_position_gain"].as<double>();
  }
  if (yaml["buff_fire_angle_error_deg"]) {
    fire_angle_error_ = yaml["buff_fire_angle_error_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_rune_radius_m"]) RUNE_RADIUS_M = yaml["buff_rune_radius_m"].as<double>();
  if (yaml["buff_small_direction"]) SMALL_BUFF_DIRECTION = yaml["buff_small_direction"].as<int>();
  if (yaml["buff_fire_full_observation_max_age_s"]) {
    BUFF_FIRE_FULL_OBSERVATION_MAX_AGE_S =
      yaml["buff_fire_full_observation_max_age_s"].as<double>();
  }

  last_fire_t_ = std::chrono::steady_clock::now();
}

void Aimer::reset()
{
  predicted_target_.reset();
  solution_converged_ = false;
  command_state_initialized_ = false;
  command_yaw_vel_ = 0.0;
  command_pitch_vel_ = 0.0;
  last_fire_t_ = std::chrono::steady_clock::now();
}

io::Command Aimer::aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp,
  double bullet_speed, bool to_now)
{
  io::Command command = {false, false, 0, 0};
  predicted_target_.reset();
  if (!target.can_control()) return command;

  // 如果子弹速度小于10，将其设为24
  if (bullet_speed < 10) bullet_speed = 24;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch, true)) {
    command.yaw = yaw;
    command.pitch = -pitch;  //世界坐标系下的pitch向上为负
    command.control = true;
  }

  if (!command.control || !solution_converged_ || !target.can_fire(now)) {
    command.shoot = false;
    last_fire_t_ = now;
  } else if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

auto_aim::Plan Aimer::mpc_aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  predicted_target_.reset();
  if (!target.can_control()) return plan;

  double bullet_speed;
  // 如果子弹速度小于10，将其设为24
  if (gs.bullet_speed < 10)
    bullet_speed = 24;
  else
    bullet_speed = gs.bullet_speed;

  auto now = std::chrono::steady_clock::now();

  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  double yaw, pitch;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch, true)) {
    const bool future_solution_converged = solution_converged_;
    plan.target_yaw = yaw;
    plan.target_pitch = pitch;

    double target_yaw_vel = 0.0;
    double target_pitch_vel = 0.0;
    auto target_for_now = target.clone();
    const double now_time = to_now ? detect_now_gap : 0.1;
    double now_yaw = 0.0;
    double now_pitch = 0.0;
    if (
      predict_time_ > 1e-4 &&
      get_send_angle(*target_for_now, now_time, bullet_speed, to_now, now_yaw, now_pitch)) {
      target_yaw_vel = tools::limit_rad(yaw - now_yaw) / predict_time_;
      target_pitch_vel = (pitch - now_pitch) / predict_time_;
    }
    solution_converged_ = future_solution_converged;
    update_command(yaw, pitch, target_yaw_vel, target_pitch_vel, gs, now, plan);
  }

  const double feedback_yaw = gs.yaw / 57.3;
  const double feedback_pitch = gs.pitch / 57.3;
  const bool command_settled =
    plan.control &&
    std::abs(tools::limit_rad(plan.target_yaw - plan.yaw)) <= fire_angle_error_ &&
    std::abs(plan.target_pitch - plan.pitch) <= fire_angle_error_;
  const bool gimbal_settled =
    plan.control && std::isfinite(feedback_yaw) && std::isfinite(feedback_pitch) &&
    std::abs(tools::limit_rad(plan.target_yaw - feedback_yaw)) <= fire_angle_error_ &&
    std::abs(plan.target_pitch - feedback_pitch) <= fire_angle_error_;

  if (
    !plan.control || !solution_converged_ || !target.can_fire(now) || !command_settled ||
    !gimbal_settled) {
    plan.fire = false;
    last_fire_t_ = now;
  } else if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
}

void Aimer::update_command(
  double target_yaw, double target_pitch, double target_yaw_vel, double target_pitch_vel,
  const io::GimbalState & gs, std::chrono::steady_clock::time_point now, auto_aim::Plan & plan)
{
  if (!command_state_initialized_) {
    const double feedback_yaw = gs.yaw / 57.3;
    const double feedback_pitch = gs.pitch / 57.3;
    command_yaw_ = std::isfinite(feedback_yaw) ? feedback_yaw : target_yaw;
    command_pitch_ = std::isfinite(feedback_pitch) ? feedback_pitch : target_pitch;
    command_yaw_vel_ = 0.0;
    command_pitch_vel_ = 0.0;
    last_command_t_ = now;
    command_state_initialized_ = true;
  }

  const double raw_dt = tools::delta_time(now, last_command_t_);
  const double dt = std::clamp(raw_dt, 0.002, 0.05);

  auto update_axis = [&](double target, double feedforward_vel, double max_vel, double max_acc,
                         bool wrap, double & position, double & velocity) {
    const double error = wrap ? tools::limit_rad(target - position) : target - position;
    const double previous_velocity = velocity;
    const double desired_velocity = std::clamp(
      feedforward_vel + command_position_gain_ * error, -max_vel, max_vel);
    const double velocity_delta =
      std::clamp(desired_velocity - velocity, -max_acc * dt, max_acc * dt);
    velocity += velocity_delta;
    double step = velocity * dt;
    if (
      std::abs(step) > std::abs(error) &&
      (step == 0.0 || error == 0.0 || std::signbit(step) == std::signbit(error))) {
      step = error;
      const double settled_velocity = std::clamp(feedforward_vel, -max_vel, max_vel);
      velocity = std::clamp(
        settled_velocity, previous_velocity - max_acc * dt, previous_velocity + max_acc * dt);
    }
    position += step;
    if (wrap) position = tools::limit_rad(position);
    return (velocity - previous_velocity) / dt;
  };

  const double yaw_acc = update_axis(
    target_yaw, target_yaw_vel, max_yaw_vel_, max_yaw_acc_, true, command_yaw_,
    command_yaw_vel_);
  const double pitch_acc = update_axis(
    target_pitch, target_pitch_vel, max_pitch_vel_, max_pitch_acc_, false, command_pitch_,
    command_pitch_vel_);

  last_command_t_ = now;
  plan.control = true;
  plan.yaw = command_yaw_;
  plan.yaw_vel = command_yaw_vel_;
  plan.yaw_acc = yaw_acc;
  plan.pitch = command_pitch_;
  plan.pitch_vel = command_pitch_vel_;
  plan.pitch_acc = pitch_acc;
}

bool Aimer::get_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  const bool to_now, double & yaw, double & pitch, bool save_prediction)
{
  (void)to_now;
  solution_converged_ = false;
  double flight_time = 0.0;
  bool has_solution = false;
  std::unique_ptr<Target> solved_prediction;

  for (int iteration = 0; iteration < 3; ++iteration) {
    auto predicted = target.clone();
    predicted->predict(std::max(0.0, predict_time + flight_time));
    angle = predicted->ekf_x()[5];

    const auto aim_in_world =
      predicted->point_buff2world(Eigen::Vector3d(0.0, 0.0, RUNE_RADIUS_M));
    const double d = std::sqrt(
      aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
    const double h = aim_in_world[2];
    if (!std::isfinite(d) || !std::isfinite(h) || d > 20.0) {
      tools::logger()->debug("[Aimer] Invalid coordinate blocked: d={:.2f}, h={:.2f}", d, h);
      return false;
    }

    tools::Trajectory trajectory(bullet_speed, d, h);
    if (trajectory.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory: {:.2f} {:.2f} {:.2f}", bullet_speed, d, h);
      return false;
    }

    yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
    pitch = trajectory.pitch + pitch_offset_;
    has_solution = true;
    const double time_error = trajectory.fly_time - flight_time;
    flight_time = trajectory.fly_time;
    if (save_prediction) solved_prediction = std::move(predicted);
    if (iteration > 0 && std::abs(time_error) <= 0.005) {
      solution_converged_ = true;
      break;
    }
  }

  if (has_solution && !solution_converged_) {
    tools::logger()->debug("[Aimer] trajectory did not converge, control only");
  }
  if (has_solution && save_prediction) predicted_target_ = std::move(solved_prediction);
  return has_solution;
};

}  // namespace auto_buff
