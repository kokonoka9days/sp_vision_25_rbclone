#ifndef AUTO_DRONE__SOLVER_HPP
#define AUTO_DRONE__SOLVER_HPP

#include <Eigen/Dense>  
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "drone_armor.hpp"

namespace auto_drone
{
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // 注意：参数原名遗留叫Armor，这里也改为 Drone 增加统一性
  void solve(Drone & drone) const;

  std::vector<cv::Point2f> reproject_drone(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Vector3d & ypr_in_world) const;

  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__SOLVER_HPP