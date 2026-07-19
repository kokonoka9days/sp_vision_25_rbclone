#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{

struct CameraContext
{
  Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 5, 1> distortion = Eigen::Matrix<double, 5, 1>::Zero();
  Eigen::Isometry3d T_camera_world = Eigen::Isometry3d::Identity();
};

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;
  void set_R_gimbal2world(const Eigen::Quaterniond & q);
  CameraContext camera_context() const;

  bool solve(Armor & armor) const;

  void omn_dig_yaw_solve(
    Armor & armor, Eigen::Vector3d R_camera2biggimbal_ypr,
    Eigen::Vector3d t_camera2biggimbal) const;

  std::vector<cv::Point2f> reproject_pose(
    const Eigen::Isometry3d & pose_in_world, ArmorType type) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  double oupost_reprojection_error(Armor armor, const double & pitch) const;
  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & world_points) const;

private:
  cv::Mat camera_matrix_cv_;
  cv::Mat distortion_cv_;
  Eigen::Matrix3d camera_matrix_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 5, 1> distortion_ = Eigen::Matrix<double, 5, 1>::Zero();
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();

  double armor_reprojection_error(const Armor & armor, const Eigen::Isometry3d & pose) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
