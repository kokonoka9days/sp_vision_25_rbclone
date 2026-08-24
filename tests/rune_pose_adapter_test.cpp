#include <cmath>
#include <cstdlib>

#include <Eigen/Geometry>

#include "pose_adapter.hpp"

#define CHECK(condition) do { if (!(condition)) std::abort(); } while (false)

int main()
{
  const auto pose = auto_buff::detail::convert_world_camera_to_rp(
    Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 2.0, 3.0));
  const Eigen::Vector3d expected_translation(-2.0, -3.0, 1.0);
  CHECK((pose.t_car_from_camera - expected_translation).norm() < 1e-12);

  CHECK((pose.R_car_from_camera * Eigen::Vector3d::UnitX() -
         Eigen::Vector3d::UnitZ()).norm() < 1e-12);
  CHECK((pose.R_car_from_camera * Eigen::Vector3d::UnitY() +
         Eigen::Vector3d::UnitX()).norm() < 1e-12);
  CHECK((pose.R_car_from_camera * Eigen::Vector3d::UnitZ() +
         Eigen::Vector3d::UnitY()).norm() < 1e-12);

  const Eigen::Vector3d current_target(5.0, 1.0, 0.5);
  const Eigen::Vector3d rp_target(-current_target.y(), -current_target.z(), current_target.x());
  const double current_yaw = std::atan2(current_target.y(), current_target.x());
  const double rp_yaw = std::atan2(rp_target.x(), rp_target.z());
  const double current_pitch = std::atan2(
    current_target.z(), std::hypot(current_target.x(), current_target.y()));
  const double rp_pitch = std::atan2(-rp_target.y(), std::hypot(rp_target.x(), rp_target.z()));
  CHECK(std::abs(current_yaw + rp_yaw) < 1e-12);
  CHECK(std::abs(current_pitch - rp_pitch) < 1e-12);
  return 0;
}
