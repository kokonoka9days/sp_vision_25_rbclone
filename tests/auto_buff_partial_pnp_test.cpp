#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/buff_solver.hpp"

namespace
{
cv::Point2f mean_point(const std::vector<cv::Point2f> & points, size_t begin, size_t end)
{
  cv::Point2f sum{0.0f, 0.0f};
  for (size_t i = begin; i < end; ++i) sum += points[i];
  return sum * (1.0f / static_cast<float>(end - begin));
}

auto_buff::BuffObservation make_observation(
  auto_buff::BuffObservationType type, auto_buff::RuneCenterSource center_source,
  const std::vector<cv::Point2f> & projected)
{
  auto_buff::BuffObservation observation;
  observation.type = type;
  observation.center_source = center_source;
  observation.r_center = projected[8];
  observation.timestamp = std::chrono::steady_clock::now();
  observation.track_id = 1;
  observation.confidence = 0.95f;

  const auto target_center = projected[9];
  const auto fan_center = mean_point(projected, 4, 8);
  if (type != auto_buff::BuffObservationType::FAN_ONLY) {
    observation.target_points.assign(projected.begin(), projected.begin() + 4);
    observation.target_center = target_center;
    observation.target_center_observed = true;
  }
  if (type != auto_buff::BuffObservationType::TARGET_ONLY) {
    observation.fan_points.assign(projected.begin() + 4, projected.begin() + 8);
    observation.fan_center = fan_center;
    observation.fan_center_observed = true;
  }
  const auto angle_point =
    type == auto_buff::BuffObservationType::FAN_ONLY ? fan_center : target_center;
  observation.angle = std::atan2(
    angle_point.y - observation.r_center.y, angle_point.x - observation.r_center.x);
  return observation;
}

bool require(bool condition, const char * message)
{
  if (condition) return true;
  std::cerr << "FAILED: " << message << '\n';
  return false;
}
}  // namespace

int main(int argc, char ** argv)
{
  const std::string config = argc > 1 ? argv[1] : "../configs/xiaohei.yaml";
  const auto yaml = YAML::LoadFile(config);
  const auto camera_data = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distortion_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_eigen(camera_data.data());
  Eigen::Matrix<double, 1, 5> distortion_eigen(distortion_data.data());
  cv::Mat camera_matrix;
  cv::Mat distortion;
  cv::eigen2cv(camera_eigen, camera_matrix);
  cv::eigen2cv(distortion_eigen, distortion);

  const float radius = static_cast<float>(auto_buff::RUNE_RADIUS_M);
  std::vector<cv::Point3f> object_points{
    {0.0f, -0.095f, 0.0f}, {0.095f, 0.0f, 0.0f}, {0.0f, 0.095f, 0.0f},
    {-0.095f, 0.0f, 0.0f}, {-0.030f, 0.191f, 0.0f}, {0.030f, 0.191f, 0.0f},
    {0.030f, 0.521f, 0.0f}, {-0.030f, 0.521f, 0.0f}, {0.0f, radius, 0.0f},
    {0.0f, 0.0f, 0.0f}};
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    object_points, cv::Vec3d(0.08, -0.12, 0.15), cv::Vec3d(0.10, -0.05, 5.0),
    camera_matrix, distortion, projected);

  auto_buff::Solver solver(config);
  const auto full = solver.solve(make_observation(
    auto_buff::BuffObservationType::FULL, auto_buff::RuneCenterSource::DETECTED, projected));
  if (!require(full.has_value(), "full 8-point PnP")) return 1;
  if (!require(
        full->pose_quality == auto_buff::BuffPoseQuality::FULL_8_POINT,
        "full pose quality")) {
    return 1;
  }

  const auto target_five = solver.solve(make_observation(
    auto_buff::BuffObservationType::TARGET_ONLY, auto_buff::RuneCenterSource::DETECTED,
    projected));
  if (!require(target_five.has_value(), "target + center 5-point PnP")) return 1;
  if (!require(
        target_five->pose_quality == auto_buff::BuffPoseQuality::PARTIAL_5_POINT,
        "target 5-point pose quality")) {
    return 1;
  }

  const auto fan_five = solver.solve(make_observation(
    auto_buff::BuffObservationType::FAN_ONLY, auto_buff::RuneCenterSource::DETECTED, projected));
  if (!require(fan_five.has_value(), "fan + center 5-point PnP")) return 1;
  if (!require(
        fan_five->pose_quality == auto_buff::BuffPoseQuality::PARTIAL_5_POINT,
        "fan 5-point pose quality")) {
    return 1;
  }

  const auto target_four = solver.solve(make_observation(
    auto_buff::BuffObservationType::TARGET_ONLY, auto_buff::RuneCenterSource::PREDICTED,
    projected));
  if (!require(target_four.has_value(), "target-only 4-point PnP")) return 1;
  if (!require(
        target_four->pose_quality == auto_buff::BuffPoseQuality::PARTIAL_4_POINT,
        "target 4-point pose quality")) {
    return 1;
  }

  const auto fan_four = solver.solve(make_observation(
    auto_buff::BuffObservationType::FAN_ONLY, auto_buff::RuneCenterSource::PREDICTED,
    projected));
  if (!require(fan_four.has_value(), "fan-only 4-point PnP")) return 1;
  if (!require(
        fan_four->pose_quality == auto_buff::BuffPoseQuality::PARTIAL_4_POINT,
        "fan 4-point pose quality")) {
    return 1;
  }

  auto bad_center = make_observation(
    auto_buff::BuffObservationType::TARGET_ONLY, auto_buff::RuneCenterSource::PREDICTED,
    projected);
  bad_center.r_center += cv::Point2f(100.0f, 0.0f);
  if (!require(!solver.solve(bad_center).has_value(), "bad predicted center rejection")) return 1;

  std::cout << "auto_buff_partial_pnp_test passed\n";
  return 0;
}
