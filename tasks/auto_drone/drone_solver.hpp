#ifndef AUTO_DRONE__SOLVER_HPP
#define AUTO_DRONE__SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "drone_armor.hpp"

namespace auto_drone
{
struct SolveDiagnostics
{
  bool valid_input = false;
  bool pnp_success = false;
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  Eigen::Matrix3d R_object2camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d model_center_object = Eigen::Vector3d::Zero();
  Eigen::Vector3d origin_camera = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_camera = Eigen::Vector3d::Zero();
  Eigen::Vector3d origin_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_gimbal = Eigen::Vector3d::Zero();
  Eigen::Vector3d origin_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
  cv::Point2f keypoint_center{};
  cv::Point2f principal_point{};
  cv::Point2f origin_pixel{};
  cv::Point2f center_pixel{};
  cv::Point2f roundtrip_center_pixel{};
  std::vector<cv::Point2f> reprojected_points;
  double reprojection_rmse_px = 0.0;
  double reprojection_max_px = 0.0;
  double roundtrip_error_px = 0.0;
};

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // 注意：参数原名遗留叫Armor，这里也改为 Drone 增加统一性
  void solve(Drone & drone) const;
  SolveDiagnostics diagnose(const Drone & drone) const;
  Eigen::Vector3d pixel_ray(const cv::Point2f & pixel) const;
  cv::Point2f principal_point() const;

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
