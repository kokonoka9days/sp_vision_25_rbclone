#include "planner.hpp"

#include "tools/math_tools.hpp"

namespace auto_aim
{
Trajectory Planner::get_trajectory(Target target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);
    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }
  return traj;
}

Trajectory Planner::rbget_trajectory(Target target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = rbaim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = rbaim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = rbaim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);
    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }
  return traj;
}

Trajectory Planner::rbget_trajectory_split(
  Target yaw_target, Target pitch_target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  yaw_target.predict(-DT * (HALF_HORIZON + 1));
  pitch_target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = rbaim(yaw_target, bullet_speed);
  auto pitch_pitch_last = rbaim(pitch_target, bullet_speed);

  yaw_target.predict(DT);
  pitch_target.predict(DT);
  auto yaw_pitch = rbaim(yaw_target, bullet_speed);
  auto pitch_pitch = rbaim(pitch_target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    yaw_target.predict(DT);
    pitch_target.predict(DT);
    auto yaw_pitch_next = rbaim(yaw_target, bullet_speed);
    auto pitch_pitch_next = rbaim(pitch_target, bullet_speed);

    const auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    const auto pitch_vel = (pitch_pitch_next(1) - pitch_pitch_last(1)) / (2 * DT);
    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, pitch_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
    pitch_pitch_last = pitch_pitch;
    pitch_pitch = pitch_pitch_next;
  }
  return traj;
}
}  // namespace auto_aim
