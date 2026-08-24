#ifndef AUTO_BUFF__POSE_ADAPTER_HPP
#define AUTO_BUFF__POSE_ADAPTER_HPP

#include <Eigen/Geometry>

#include "runtime_types.hpp"

namespace auto_buff::detail
{
CameraPose convert_world_camera_to_rp(
  const Eigen::Matrix3d & R_camera_to_world,
  const Eigen::Vector3d & t_camera_in_world);
CameraPose make_rp_camera_pose(const Eigen::Quaterniond & imu_q);
}  // namespace auto_buff::detail

#endif
