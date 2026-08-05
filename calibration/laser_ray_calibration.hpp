#ifndef CALIBRATION__LASER_RAY_CALIBRATION_HPP
#define CALIBRATION__LASER_RAY_CALIBRATION_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

namespace laser_calibration
{

struct CameraModel
{
  cv::Mat camera_matrix;
  cv::Mat distort_coeffs;
};

struct BoardConfig
{
  int pattern_cols = 0;
  int pattern_rows = 0;
  double square_size_m = 0.0;
};

struct RawSample
{
  int id = 0;
  std::string image_path;
  cv::Point2f laser_pixel{};
  cv::Size image_size{};
  int64_t timestamp_ns = 0;
};

struct BoardObservation
{
  bool valid = false;
  std::string rejection_reason;
  std::vector<cv::Point2f> corners;
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  Eigen::Vector3d plane_normal_camera = Eigen::Vector3d::Zero();
  double plane_offset_camera = 0.0;
  double reprojection_rmse_px = std::numeric_limits<double>::quiet_NaN();
};

struct ProcessedSample
{
  RawSample raw;
  BoardObservation board;
  bool valid = false;
  std::string rejection_reason;
  Eigen::Vector3d point_camera =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  bool ransac_inlier = false;
  bool validation_sample = false;
  double point_line_error_m = std::numeric_limits<double>::quiet_NaN();
  cv::Point2f predicted_pixel{
    std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
  double pixel_error_px = std::numeric_limits<double>::quiet_NaN();
};

struct LaserLine
{
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
};

struct ErrorStats
{
  int count = 0;
  double rms = std::numeric_limits<double>::quiet_NaN();
  double p95 = std::numeric_limits<double>::quiet_NaN();
  double maximum = std::numeric_limits<double>::quiet_NaN();
};

struct FitOptions
{
  double max_board_reprojection_rmse_px = 1.5;
  double ransac_threshold_m = 0.005;
  int ransac_iterations = 2000;
  uint32_t random_seed = 0x5A17U;
  int bootstrap_iterations = 200;
  double huber_loss_px = 2.0;
  int min_valid_samples = 15;
  int min_samples_per_distance_bin = 3;
  double min_sample_span_m = 0.5;
  double min_inlier_ratio = 0.8;
  double max_point_line_rms_m = 0.005;
  double max_validation_rms_px = 2.0;
  double max_validation_p95_px = 4.0;
  double max_bootstrap_direction_p95_deg = 0.1;
};

struct FitResult
{
  bool fit_succeeded = false;
  bool quality_passed = false;
  LaserLine line;
  std::vector<ProcessedSample> samples;
  std::vector<std::string> failure_reasons;
  int valid_sample_count = 0;
  int inlier_count = 0;
  int training_count = 0;
  int validation_count = 0;
  int distance_bin_counts[3] = {0, 0, 0};
  double inlier_ratio = 0.0;
  double sample_span_m = 0.0;
  ErrorStats point_line_error_m;
  ErrorStats training_pixel_error_px;
  ErrorStats validation_pixel_error_px;
  double bootstrap_direction_p95_deg = std::numeric_limits<double>::quiet_NaN();
  double bootstrap_origin_p95_m = std::numeric_limits<double>::quiet_NaN();
  std::string ceres_summary;
};

std::vector<cv::Point3f> board_object_points(const BoardConfig & config);

BoardObservation observe_board(
  const cv::Mat & image, const CameraModel & camera, const BoardConfig & board);

std::optional<Eigen::Vector3d> intersect_pixel_with_plane(
  const cv::Point2f & pixel, const CameraModel & camera, const Eigen::Vector3d & plane_normal,
  double plane_offset);

ProcessedSample process_sample(
  const RawSample & sample, const cv::Mat & image, const CameraModel & camera,
  const BoardConfig & board, const FitOptions & options);

std::optional<cv::Point2f> project_line_to_board(
  const LaserLine & line, const BoardObservation & board, const CameraModel & camera,
  Eigen::Vector3d * intersection_camera = nullptr);

FitResult fit_laser_ray(
  std::vector<ProcessedSample> samples, const CameraModel & camera, const FitOptions & options);

double point_line_distance(const Eigen::Vector3d & point, const LaserLine & line);
double line_direction_error_deg(const LaserLine & lhs, const LaserLine & rhs);

}  // namespace laser_calibration

#endif  // CALIBRATION__LASER_RAY_CALIBRATION_HPP
