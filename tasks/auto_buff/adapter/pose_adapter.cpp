#include "pose_adapter.hpp"

#include "json.hpp"

namespace auto_buff::detail
{
CameraPose convert_world_camera_to_rp(
  const Eigen::Matrix3d & R_camera_to_world,
  const Eigen::Vector3d & t_camera_in_world)
{
  Eigen::Matrix3d world_to_rp;
  world_to_rp << 0.0, -1.0, 0.0,
                 0.0, 0.0, -1.0,
                 1.0, 0.0, 0.0;
  return {world_to_rp * R_camera_to_world, world_to_rp * t_camera_in_world};
}

CameraPose make_rp_camera_pose(const Eigen::Quaterniond & imu_q)
{
  const Eigen::Matrix3d &R_gimbal2imubody = J_POWER_RUNE.R_gimbal2imubody();
  const Eigen::Matrix3d R_gimbal2world =
    R_gimbal2imubody.transpose() * imu_q.normalized().toRotationMatrix() * R_gimbal2imubody;
  return convert_world_camera_to_rp(
    R_gimbal2world * J_POWER_RUNE.R_camera2gimbal(),
    R_gimbal2world * J_POWER_RUNE.t_camera2gimbal());
}
}  // namespace auto_buff::detail
