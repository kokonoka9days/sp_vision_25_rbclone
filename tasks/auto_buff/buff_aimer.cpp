#include "buff_aimer.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{
Aimer::Aimer(const std::string & config_path)
: Aimer(config_path, load_buff_config(config_path))
{
}

Aimer::Aimer(const std::string & config_path, BuffConfig config)
: config_(std::move(config))
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;      // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;  // degree to rad
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();

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
  // 用 {} 而非逐个列举：Plan 有 13 个成员，旧写法只给了 10 个初始化器，
  // 以后往 Plan 加字段会静默改变各初始化器的对应关系
  auto_aim::Plan plan{};
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
    plan.yaw = yaw;
    plan.pitch = pitch;
    plan.control = true;

    if (first_in_aimer_) {
      plan.yaw_vel = 0; plan.yaw_acc = 0;
      plan.pitch_vel = 0; plan.pitch_acc = 0;
      first_in_aimer_ = false;
    } else {
      // 前馈速度/加速度用「同一个预测器、同一套符号、同一个单位」的三点中心差分求得。
      // 三个采样点围绕实际下发的那一点 (future) 对称展开，间隔均为 predict_time_：
      //   prev = solve(now_time)                     -> 命令点的前一点
      //   cur  = solve(future)          = yaw/pitch   -> 实际下发的命令
      //   next = solve(future + dt)                  -> 命令点的后一点
      // 不能像以前那样拿 gs.yaw / gs.pitch 当中间点：它们是云台「实测」角且单位是度
      // (gimbal.cpp 里 state_.yaw = ypr[0] * 57.3)，与弧度的解算结果混算无物理意义。
      const double dt = predict_time_;
      const double now_time = to_now ? detect_now_gap : 0.1;
      double prev_yaw, prev_pitch, next_yaw, next_pitch;

      // 使用全新的副本推算各时刻位姿，而不是传入负时间！
      auto target_prev = target.clone();
      auto target_next = target.clone();
      const bool prev_ok =
        get_send_angle(*target_prev, now_time, bullet_speed, to_now, prev_yaw, prev_pitch);
      const bool next_ok =
        get_send_angle(*target_next, future + dt, bullet_speed, to_now, next_yaw, next_pitch);

      if (prev_ok && next_ok && dt > 0) {
        // yaw 需要考虑 ±π 环绕，逐段取最短弧后再做差分
        const double yaw_back = tools::limit_rad(yaw - prev_yaw);
        const double yaw_fwd = tools::limit_rad(next_yaw - yaw);
        plan.yaw_vel = (yaw_fwd + yaw_back) / (2 * dt);
        plan.yaw_acc = (yaw_fwd - yaw_back) / (dt * dt);

        // pitch 不会环绕，且与 plan.pitch 同为「向上为正」，直接差分
        const double pitch_back = pitch - prev_pitch;
        const double pitch_fwd = next_pitch - pitch;
        plan.pitch_vel = (pitch_fwd + pitch_back) / (2 * dt);
        plan.pitch_acc = (pitch_fwd - pitch_back) / (dt * dt);
      } else {
        plan.yaw_vel = plan.yaw_acc = 0.0;
        plan.pitch_vel = plan.pitch_acc = 0.0;
      }
    }
    // get_send_angle 会覆写 solution_converged_，此处恢复为下发解的收敛状态
    solution_converged_ = future_solution_converged;
  }

  if (!plan.control || !solution_converged_ || !target.can_fire(now)) {
    plan.fire = false;
    last_fire_t_ = now;
  } else if (tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
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
      predicted->point_buff2world(Eigen::Vector3d(0.0, 0.0, config_.rune_radius_m));
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
