#include "solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{

std::vector<cv::Point3f> cv_armor_object_points(ArmorType type)
{
  std::vector<cv::Point3f> result;
  result.reserve(4);
  for (const auto & point : armor_object_points(type)) {
    result.emplace_back(point.x(), point.y(), point.z());
  }
  return result;
}

bool valid_image_points(const std::vector<cv::Point2f> & points)
{
  if (
    points.size() != 4 ||
    !std::all_of(points.begin(), points.end(), [](const auto & point) {
      return std::isfinite(point.x) && std::isfinite(point.y);
    })) {
    return false;
  }
  double twice_area = 0;
  for (int i = 0; i < 4; ++i) {
    const int next = (i + 1) % 4;
    if (cv::norm(points[i] - points[next]) <= 1e-3) return false;
    twice_area += points[i].x * points[next].y - points[next].x * points[i].y;
  }
  return std::abs(twice_area) > 1e-3;
}

Eigen::Matrix3d armor_rotation(double yaw, double pitch)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
}

}  // namespace

Solver::Solver(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  const auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  const auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  const auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distortion_data = yaml["distort_coeffs"].as<std::vector<double>>();

  R_gimbal2imubody_ =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Map<const Eigen::Vector3d>(t_camera2gimbal_data.data());
  camera_matrix_ =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(camera_matrix_data.data());
  distortion_ = Eigen::Map<const Eigen::Matrix<double, 5, 1>>(distortion_data.data());
  cv::eigen2cv(camera_matrix_, camera_matrix_cv_);
  const Eigen::Matrix<double, 1, 5> distortion_row = distortion_.transpose();
  cv::eigen2cv(distortion_row, distortion_cv_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  const Eigen::Matrix3d R_imubody2imuabs = q.normalized().toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

CameraContext Solver::camera_context() const
{
  CameraContext context;
  context.camera_matrix = camera_matrix_;
  context.distortion = distortion_;
  context.T_camera_world.linear() = R_gimbal2world_ * R_camera2gimbal_;
  context.T_camera_world.translation() = R_gimbal2world_ * t_camera2gimbal_;
  return context;
}

bool Solver::solve(Armor & armor) const
{
  armor.pnp_valid = false;
  armor.pose_in_world = Eigen::Isometry3d::Identity();
  armor.xyz_in_gimbal.setZero();
  armor.xyz_in_world.setZero();
  armor.ypr_in_gimbal.setZero();
  armor.ypr_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.yaw_raw = 0;
  if (!valid_image_points(armor.points)) return false;

  std::vector<cv::Mat> rotation_vectors;
  std::vector<cv::Mat> translation_vectors;
  cv::Mat reprojection_errors;
  const bool solved = cv::solvePnPGeneric(
    cv_armor_object_points(armor.type), armor.points, camera_matrix_cv_, distortion_cv_,
    rotation_vectors, translation_vectors, false, cv::SOLVEPNP_IPPE, cv::noArray(),
    cv::noArray(), reprojection_errors);
  if (!solved || rotation_vectors.empty()) return false;

  double best_error = std::numeric_limits<double>::infinity();
  Eigen::Isometry3d best_pose_in_camera = Eigen::Isometry3d::Identity();
  bool found = false;
  for (std::size_t i = 0; i < rotation_vectors.size(); ++i) {
    cv::Mat rotation_cv;
    cv::Rodrigues(rotation_vectors[i], rotation_cv);
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    cv::cv2eigen(rotation_cv, rotation);
    cv::cv2eigen(translation_vectors[i], translation);
    if (!rotation.allFinite() || !translation.allFinite() || translation.z() <= 1e-6) continue;

    const Eigen::Vector3d front_normal = -rotation.col(0);
    if (front_normal.dot(-translation) <= 0) continue;
    double error = static_cast<double>(i);
    if (!reprojection_errors.empty()) {
      const int row = reprojection_errors.rows == 1 ? 0 : static_cast<int>(i);
      const int column = reprojection_errors.rows == 1 ? static_cast<int>(i) : 0;
      error = reprojection_errors.depth() == CV_32F
                ? static_cast<double>(reprojection_errors.at<float>(row, column))
                : reprojection_errors.at<double>(row, column);
    }
    if (error >= best_error) continue;
    best_error = error;
    best_pose_in_camera.linear() = rotation;
    best_pose_in_camera.translation() = translation;
    found = true;
  }
  if (!found) return false;

  const auto context = camera_context();
  armor.pose_in_world = context.T_camera_world * best_pose_in_camera;
  armor.pnp_valid = true;
  armor.xyz_in_gimbal = R_camera2gimbal_ * best_pose_in_camera.translation() + t_camera2gimbal_;
  armor.xyz_in_world = armor.pose_in_world.translation();
  const Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * best_pose_in_camera.linear();
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(armor.pose_in_world.linear(), 2, 1, 0);
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  armor.yaw_raw = armor.ypr_in_world[0];
  return true;
}

void Solver::omn_dig_yaw_solve(
  Armor & armor, Eigen::Vector3d camera_to_gimbal_ypr,
  Eigen::Vector3d camera_to_gimbal_translation) const
{
  cv::Vec3d rotation_vector;
  cv::Vec3d translation_vector;
  if (!cv::solvePnP(
        cv_armor_object_points(armor.type), armor.points, camera_matrix_cv_, distortion_cv_,
        rotation_vector, translation_vector, false, cv::SOLVEPNP_IPPE))
    return;
  Eigen::Vector3d translation;
  cv::cv2eigen(translation_vector, translation);
  armor.xyz_in_gimbal =
    tools::rotation_matrix(camera_to_gimbal_ypr) * translation + camera_to_gimbal_translation;
}

std::vector<cv::Point2f> Solver::reproject_pose(
  const Eigen::Isometry3d & pose_in_world, ArmorType type) const
{
  const Eigen::Isometry3d pose_in_camera = camera_context().T_camera_world.inverse() * pose_in_world;
  const auto object_points = armor_object_points(type);
  for (const auto & point : object_points) {
    const Eigen::Vector3d point_in_camera = pose_in_camera * point;
    if (!point_in_camera.allFinite() || point_in_camera.z() <= 1e-6) return {};
  }

  cv::Mat rotation_cv;
  const Eigen::Matrix3d rotation = pose_in_camera.linear();
  cv::eigen2cv(rotation, rotation_cv);
  cv::Vec3d rotation_vector;
  cv::Rodrigues(rotation_cv, rotation_vector);
  const auto & translation = pose_in_camera.translation();
  const cv::Vec3d translation_vector(translation.x(), translation.y(), translation.z());
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    cv_armor_object_points(type), rotation_vector, translation_vector, camera_matrix_cv_,
    distortion_cv_, image_points);
  if (!valid_image_points(image_points)) return {};
  return image_points;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = xyz_in_world;
  const double pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0
                                                     : 15.0 * CV_PI / 180.0;
  pose.linear() = armor_rotation(yaw, pitch);
  return reproject_pose(pose, type);
}

double Solver::armor_reprojection_error(const Armor & armor, const Eigen::Isometry3d & pose) const
{
  const auto projected = reproject_pose(pose, armor.type);
  if (projected.size() != armor.points.size()) return std::numeric_limits<double>::infinity();
  double error = 0;
  for (std::size_t i = 0; i < projected.size(); ++i) error += cv::norm(projected[i] - armor.points[i]);
  return error;
}

double Solver::oupost_reprojection_error(Armor armor, const double & pitch) const
{
  if (!solve(armor)) return std::numeric_limits<double>::infinity();
  Eigen::Isometry3d constrained = armor.pose_in_world;
  constrained.linear() = armor_rotation(armor.ypr_in_world[0], pitch);
  return armor_reprojection_error(armor, constrained);
}

std::vector<cv::Point2f> Solver::world2pixel(
  const std::vector<cv::Point3f> & world_points) const
{
  const Eigen::Isometry3d world_to_camera = camera_context().T_camera_world.inverse();
  std::vector<cv::Point2f> projected;
  projected.reserve(world_points.size());
  for (const auto & point : world_points) {
    const Eigen::Vector3d camera_point =
      world_to_camera * Eigen::Vector3d(point.x, point.y, point.z);
    if (!camera_point.allFinite() || camera_point.z() <= 1e-6) continue;
    const auto pixel = tools::project_point(camera_point, camera_matrix_, distortion_);
    projected.emplace_back(pixel.x(), pixel.y());
  }
  return projected;
}

}  // namespace auto_aim
