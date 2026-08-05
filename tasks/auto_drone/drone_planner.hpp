#ifndef AUTO_DRONE__DRONE_PLANNER_HPP
#define AUTO_DRONE__DRONE_PLANNER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>

#include "drone_target.hpp"
#include "laser_ray_aim.hpp"
#include "tinympc/tiny_api.hpp"
#include "tools/logger.hpp"

namespace auto_drone
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0.0F;
  float target_pitch = 0.0F;
  float yaw = 0.0F;
  float yaw_vel = 0.0F;
  float yaw_acc = 0.0F;
  float pitch = 0.0F;
  float pitch_vel = 0.0F;
  float pitch_acc = 0.0F;
};

struct PlanDiagnostics
{
  bool target_present = false;
  bool timestamps_valid = false;
  bool target_age_valid = false;
  bool aim_valid = false;
  bool plan_valid = false;
  double observation_age_s = 0.0;
  double state_age_s = 0.0;
  double prediction_horizon_s = 0.0;
  std::chrono::steady_clock::time_point prediction_target_timestamp{};
  Eigen::Vector3d input_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d input_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector2d input_yaw_pitch = Eigen::Vector2d::Zero();
  Eigen::Vector2d predicted_yaw_pitch = Eigen::Vector2d::Zero();
  Plan plan;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  double aim_target_yaw;

  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  PlanDiagnostics plan_diagnostics(std::optional<Target> target, double bullet_speed);

  void adjust_aim_offset(double yaw_delta_deg, double pitch_delta_deg);
  double yaw_offset_deg() const;
  double pitch_offset_deg() const;

  inline Plan plan(std::optional<Target> target, double bullet_speed)
  {
    if (!target.has_value()) return {false};

    const auto now = std::chrono::steady_clock::now();
    if (
      target->state_timestamp() == std::chrono::steady_clock::time_point{} ||
      target->last_observation_timestamp() == std::chrono::steady_clock::time_point{}) {
      return {false};
    }

    const double observation_age =
      std::chrono::duration<double>(now - target->last_observation_timestamp()).count();
    if (observation_age < 0.0 || observation_age > max_target_age_) return {false};

    const double state_age =
      std::max(0.0, std::chrono::duration<double>(now - target->state_timestamp()).count());
    const double delay_time = state_age + gimbal_control_delay;

    target->predict(delay_time);

    return plan(*target, bullet_speed);
  }

private:
  std::atomic<double> yaw_offset_;
  std::atomic<double> pitch_offset_;
  double fire_thresh_;
  double gimbal_control_delay = 0.04;
  double max_target_age_ = 0.2;
  Eigen::Vector3d xyz_offset_;
  bool laser_ray_enabled_ = false;
  LaserRay laser_ray_;

  TinySolver * yaw_solver_ = nullptr;
  TinySolver * pitch_solver_ = nullptr;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  // 针对单点无人机的目标瞄准与弹道计算
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  // 轨迹生成
  Trajectory get_trajectory(Target target, double yaw0, double bullet_speed);
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__DRONE_PLANNER_HPP
