#include "calibration/laser_ray_calibration.hpp"

#include <fmt/core.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace
{

using laser_calibration::BoardObservation;
using laser_calibration::CameraModel;
using laser_calibration::FitOptions;
using laser_calibration::LaserLine;
using laser_calibration::ProcessedSample;

LaserLine canonical_line(const Eigen::Vector3d & point, Eigen::Vector3d direction)
{
  direction.normalize();
  LaserLine line;
  line.direction = direction;
  line.origin = point - direction * direction.dot(point);
  return line;
}

cv::Point2f project_point(const Eigen::Vector3d & point, const CameraModel & camera)
{
  const std::vector<cv::Point3f> points{cv::Point3f(
    static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z()))};
  std::vector<cv::Point2f> pixels;
  cv::projectPoints(
    points, cv::Vec3d::all(0), cv::Vec3d::all(0), camera.camera_matrix, camera.distort_coeffs,
    pixels);
  return pixels.front();
}

bool contains_reason(const std::vector<std::string> & reasons, const std::string & expected)
{
  return std::find(reasons.begin(), reasons.end(), expected) != reasons.end();
}

bool expect(bool condition, const std::string & message)
{
  if (!condition) fmt::print(stderr, "[FAIL] {}\n", message);
  return condition;
}

}  // namespace

int main()
{
  bool passed = true;
  CameraModel camera;
  camera.camera_matrix =
    (cv::Mat_<double>(3, 3) << 1200.0, 0.0, 640.0, 0.0, 1190.0, 480.0, 0.0, 0.0, 1.0);
  camera.distort_coeffs = (cv::Mat_<double>(1, 5) << -0.02, 0.003, 0.0005, -0.0003, 0.0);

  const LaserLine truth =
    canonical_line(Eigen::Vector3d(0.045, -0.030, 0.20), Eigen::Vector3d(0.030, -0.015, 1.0));
  std::mt19937 generator(20260718U);
  std::normal_distribution<double> pixel_noise(0.0, 0.12);
  std::vector<ProcessedSample> samples;
  for (int i = 0; i < 30; ++i) {
    const double distance = 0.8 + 0.11 * i;
    const Eigen::Vector3d point = truth.origin + distance * truth.direction;
    const double tilt_x = -0.25 + 0.50 * (i % 7) / 6.0;
    const double tilt_y = -0.20 + 0.40 * (i % 5) / 4.0;
    Eigen::Vector3d normal(std::sin(tilt_x), std::sin(tilt_y), 1.0);
    normal.normalize();

    ProcessedSample sample;
    sample.raw.id = i + 1;
    sample.raw.image_path = fmt::format("synthetic/{:04d}.jpg", i + 1);
    sample.raw.image_size = {1280, 960};
    sample.board.valid = true;
    sample.board.reprojection_rmse_px = 0.15;
    sample.board.plane_normal_camera = normal;
    sample.board.plane_offset_camera = -normal.dot(point);
    sample.raw.laser_pixel = project_point(point, camera);
    sample.raw.laser_pixel.x += static_cast<float>(pixel_noise(generator));
    sample.raw.laser_pixel.y += static_cast<float>(pixel_noise(generator));
    if (i == 2 || i == 11 || i == 17) {
      sample.raw.laser_pixel += cv::Point2f(28.0F, -24.0F);
    }
    const auto reconstructed = laser_calibration::intersect_pixel_with_plane(
      sample.raw.laser_pixel, camera, normal, sample.board.plane_offset_camera);
    passed &= expect(reconstructed.has_value(), "synthetic point must intersect its board plane");
    if (!reconstructed) return 1;
    sample.point_camera = *reconstructed;
    sample.valid = true;
    samples.push_back(sample);
  }

  FitOptions options;
  options.ransac_threshold_m = 0.008;
  options.max_point_line_rms_m = 0.008;
  const auto result = laser_calibration::fit_laser_ray(samples, camera, options);
  passed &= expect(result.fit_succeeded, "synthetic fit must succeed");
  passed &= expect(result.quality_passed, "synthetic fit must pass quality gates");
  const double direction_error = laser_calibration::line_direction_error_deg(result.line, truth);
  const double origin_error = (result.line.origin - truth.origin).norm();
  passed &= expect(direction_error < 0.08, "direction error must be below 0.08 degree");
  passed &= expect(origin_error < 0.008, "canonical line origin error must be below 8 mm");
  passed &= expect(result.inlier_count >= 26, "RANSAC must retain clean observations");
  passed &= expect(result.inlier_count < 30, "RANSAC must reject injected outliers");

  const auto repeated = laser_calibration::fit_laser_ray(samples, camera, options);
  passed &= expect(
    laser_calibration::line_direction_error_deg(result.line, repeated.line) < 1e-9,
    "fixed seed must make repeated fit deterministic");
  passed &= expect(
    (result.line.origin - repeated.line.origin).norm() < 1e-12,
    "fixed seed must make repeated origin deterministic");

  std::vector<ProcessedSample> too_few(samples.begin(), samples.begin() + 8);
  const auto insufficient = laser_calibration::fit_laser_ray(too_few, camera, options);
  passed &= expect(!insufficient.quality_passed, "too few samples must fail quality gate");
  passed &= expect(
    contains_reason(insufficient.failure_reasons, "insufficient_valid_samples"),
    "insufficient sample reason must be reported");

  const cv::Point2f principal_point(640.0F, 480.0F);
  const auto parallel = laser_calibration::intersect_pixel_with_plane(
    principal_point, camera, Eigen::Vector3d::UnitX(), -1.0);
  passed &= expect(!parallel.has_value(), "parallel camera ray and plane must be rejected");
  const auto behind = laser_calibration::intersect_pixel_with_plane(
    principal_point, camera, Eigen::Vector3d::UnitZ(), 1.0);
  passed &= expect(!behind.has_value(), "intersection behind camera must be rejected");

  laser_calibration::RawSample wrong_size;
  wrong_size.id = 1;
  wrong_size.image_size = {800, 600};
  wrong_size.laser_pixel = principal_point;
  const cv::Mat wrong_image(480, 640, CV_8UC3, cv::Scalar::all(0));
  laser_calibration::BoardConfig board;
  board.pattern_cols = 11;
  board.pattern_rows = 8;
  board.square_size_m = 0.03;
  const auto mismatched =
    laser_calibration::process_sample(wrong_size, wrong_image, camera, board, options);
  passed &= expect(
    mismatched.rejection_reason == "image_size_mismatch",
    "wrong image size must be rejected before board detection");

  laser_calibration::RawSample blank_sample;
  blank_sample.id = 2;
  blank_sample.image_size = wrong_image.size();
  blank_sample.laser_pixel = {320.0F, 240.0F};
  const auto board_missing =
    laser_calibration::process_sample(blank_sample, wrong_image, camera, board, options);
  passed &= expect(
    board_missing.rejection_reason == "board_not_found", "missing chessboard must be reported");

  blank_sample.laser_pixel = {-1.0F, 240.0F};
  const auto outside =
    laser_calibration::process_sample(blank_sample, wrong_image, camera, board, options);
  passed &= expect(
    outside.rejection_reason == "laser_pixel_outside_image",
    "out-of-bounds laser click must be rejected");

  BoardObservation parallel_board;
  parallel_board.valid = true;
  parallel_board.plane_normal_camera = Eigen::Vector3d::UnitX();
  parallel_board.plane_offset_camera = -1.0;
  LaserLine parallel_line;
  parallel_line.origin = Eigen::Vector3d::Zero();
  parallel_line.direction = Eigen::Vector3d::UnitZ();
  passed &= expect(
    !laser_calibration::project_line_to_board(parallel_line, parallel_board, camera).has_value(),
    "laser line parallel to board plane must be rejected");

  auto low_inlier_samples = samples;
  for (size_t i = 0; i < 10; ++i) {
    low_inlier_samples[i].point_camera +=
      Eigen::Vector3d(0.15 + 0.01 * i, -0.12 + 0.02 * i, 0.03 * i * i);
  }
  const auto low_inlier = laser_calibration::fit_laser_ray(low_inlier_samples, camera, options);
  passed &= expect(
    contains_reason(low_inlier.failure_reasons, "ransac_inlier_ratio_too_low"),
    "low RANSAC inlier ratio must fail quality gate");

  fmt::print(
    "direction_error_deg={:.6f} origin_error_m={:.6f} inliers={}/{} validation_rms={:.3f}px\n",
    direction_error, origin_error, result.inlier_count, result.valid_sample_count,
    result.validation_pixel_error_px.rms);
  return passed ? 0 : 1;
}
