#ifndef AUTO_DRONE__DRONE_PLANNER_HPP
#define AUTO_DRONE__DRONE_PLANNER_HPP

#include <Eigen/Dense>
#include <optional>

#include "drone_target.hpp"
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
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  double aim_target_yaw;

  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  
  inline Plan plan(std::optional<Target> target, double bullet_speed) {
    if (!target.has_value()) return {false};

    // 无人机逻辑：根据目标当前速度的模长判断高低速预测延迟
    double delay_time = gimbal_control_delay;

    target->predict(delay_time);

    return plan(*target, bullet_speed);
  }

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double gimbal_control_delay;
  Eigen::Vector3d xyz_offset_;

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