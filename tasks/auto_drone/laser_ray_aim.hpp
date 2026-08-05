#ifndef AUTO_DRONE__LASER_RAY_AIM_HPP
#define AUTO_DRONE__LASER_RAY_AIM_HPP

#include <Eigen/Dense>
#include <optional>

namespace auto_drone
{

struct LaserRay
{
  Eigen::Vector3d origin_in_gimbal_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d direction_in_gimbal = Eigen::Vector3d::UnitX();
};

struct LaserAimSolution
{
  double yaw = 0.0;
  double pitch = 0.0;
  double distance_along_ray_m = 0.0;
  double residual_m = 0.0;
};

Eigen::Matrix3d laser_command_rotation(double yaw, double pitch);

std::optional<LaserAimSolution> solve_laser_ray_aim(
  const Eigen::Vector3d & target_in_world_m, const LaserRay & laser_ray);

}  // namespace auto_drone

#endif  // AUTO_DRONE__LASER_RAY_AIM_HPP
