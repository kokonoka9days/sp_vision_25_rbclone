#include "laser_ray_calibration.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/core/eigen.hpp>
#include <random>
#include <set>
#include <sstream>

namespace laser_calibration
{
namespace
{

constexpr double kRadToDeg = 180.0 / CV_PI;

double nan_value() { return std::numeric_limits<double>::quiet_NaN(); }

std::vector<double> distortion_values(const cv::Mat & coefficients)
{
  std::vector<double> values(5, 0.0);
  const cv::Mat flattened = coefficients.reshape(1, 1);
  for (size_t i = 0; i < std::min<size_t>(5, flattened.total()); ++i) {
    values[i] = flattened.at<double>(0, static_cast<int>(i));
  }
  return values;
}

ErrorStats compute_stats(const std::vector<double> & values)
{
  ErrorStats stats;
  stats.count = static_cast<int>(values.size());
  if (values.empty()) {
    stats.rms = nan_value();
    stats.p95 = nan_value();
    stats.maximum = nan_value();
    return stats;
  }

  double squared_sum = 0.0;
  for (const double value : values) squared_sum += value * value;
  stats.rms = std::sqrt(squared_sum / values.size());
  stats.maximum = *std::max_element(values.begin(), values.end());
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const size_t index = static_cast<size_t>(std::ceil(0.95 * sorted.size())) - 1;
  stats.p95 = sorted[std::min(index, sorted.size() - 1)];
  return stats;
}

double percentile95(std::vector<double> values)
{
  if (values.empty()) return nan_value();
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(std::ceil(0.95 * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

LaserLine canonical_line(const Eigen::Vector3d & point, Eigen::Vector3d direction)
{
  direction.normalize();
  LaserLine line;
  line.direction = direction;
  line.origin = point - direction * direction.dot(point);
  return line;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> tangent_basis(const Eigen::Vector3d & direction)
{
  const Eigen::Vector3d helper =
    std::abs(direction.z()) < 0.9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitY();
  const Eigen::Vector3d first = direction.cross(helper).normalized();
  const Eigen::Vector3d second = direction.cross(first).normalized();
  return {first, second};
}

std::optional<LaserLine> pca_line(
  const std::vector<ProcessedSample> & samples, const std::vector<size_t> & indices)
{
  if (indices.size() < 2) return std::nullopt;

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const size_t index : indices) centroid += samples[index].point_camera;
  centroid /= static_cast<double>(indices.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const size_t index : indices) {
    const Eigen::Vector3d centered = samples[index].point_camera - centroid;
    covariance += centered * centered.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
  if (eigen_solver.info() != Eigen::Success) return std::nullopt;

  Eigen::Vector3d direction = eigen_solver.eigenvectors().col(2).normalized();
  LaserLine line = canonical_line(centroid, direction);
  if (line.direction.dot(centroid - line.origin) < 0.0) line.direction *= -1.0;
  return line;
}

struct ReprojectionCost
{
  ReprojectionCost(
    const LaserLine & initial, const Eigen::Vector3d & basis_first,
    const Eigen::Vector3d & basis_second, const BoardObservation & board,
    const cv::Point2f & observed_pixel, const CameraModel & camera)
  : d0(initial.direction),
    e1(basis_first),
    e2(basis_second),
    normal(board.plane_normal_camera),
    plane_offset(board.plane_offset_camera),
    observed_u(observed_pixel.x),
    observed_v(observed_pixel.y)
  {
    fx = camera.camera_matrix.at<double>(0, 0);
    skew = camera.camera_matrix.at<double>(0, 1);
    cx = camera.camera_matrix.at<double>(0, 2);
    fy = camera.camera_matrix.at<double>(1, 1);
    cy = camera.camera_matrix.at<double>(1, 2);
    const auto distortion = distortion_values(camera.distort_coeffs);
    k1 = distortion[0];
    k2 = distortion[1];
    p1 = distortion[2];
    p2 = distortion[3];
    k3 = distortion[4];
  }

  template <typename T>
  bool operator()(const T * const parameters, T * residuals) const
  {
    Eigen::Matrix<T, 3, 1> direction =
      d0.cast<T>() + parameters[0] * e1.cast<T>() + parameters[1] * e2.cast<T>();
    direction.normalize();
    Eigen::Matrix<T, 3, 1> origin = parameters[2] * e1.cast<T>() + parameters[3] * e2.cast<T>();
    origin -= direction * direction.dot(origin);

    const Eigen::Matrix<T, 3, 1> plane_normal = normal.cast<T>();
    const T denominator = plane_normal.dot(direction);
    const T lambda = -(plane_normal.dot(origin) + T(plane_offset)) / denominator;
    const Eigen::Matrix<T, 3, 1> point = origin + lambda * direction;

    const T x = point.x() / point.z();
    const T y = point.y() / point.z();
    const T r2 = x * x + y * y;
    const T radial = T(1.0) + T(k1) * r2 + T(k2) * r2 * r2 + T(k3) * r2 * r2 * r2;
    const T x_distorted = x * radial + T(2.0 * p1) * x * y + T(p2) * (r2 + T(2.0) * x * x);
    const T y_distorted = y * radial + T(p1) * (r2 + T(2.0) * y * y) + T(2.0 * p2) * x * y;
    const T projected_u = T(fx) * x_distorted + T(skew) * y_distorted + T(cx);
    const T projected_v = T(fy) * y_distorted + T(cy);

    residuals[0] = projected_u - T(observed_u);
    residuals[1] = projected_v - T(observed_v);
    return true;
  }

  Eigen::Vector3d d0;
  Eigen::Vector3d e1;
  Eigen::Vector3d e2;
  Eigen::Vector3d normal;
  double plane_offset;
  double observed_u;
  double observed_v;
  double fx;
  double skew;
  double cx;
  double fy;
  double cy;
  double k1;
  double k2;
  double p1;
  double p2;
  double k3;
};

bool refine_line(
  const std::vector<ProcessedSample> & samples, const std::vector<size_t> & indices,
  const CameraModel & camera, const FitOptions & options, LaserLine & line,
  std::string & summary_text)
{
  if (indices.size() < 2) return false;
  const auto [basis_first, basis_second] = tangent_basis(line.direction);
  double parameters[4] = {0.0, 0.0, line.origin.dot(basis_first), line.origin.dot(basis_second)};

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  auto loss = std::make_unique<ceres::HuberLoss>(options.huber_loss_px);
  int residual_count = 0;
  for (const size_t index : indices) {
    if (std::abs(samples[index].board.plane_normal_camera.dot(line.direction)) < 1e-4) continue;
    auto * cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 4>(new ReprojectionCost(
      line, basis_first, basis_second, samples[index].board, samples[index].raw.laser_pixel,
      camera));
    problem.AddResidualBlock(cost, loss.get(), parameters);
    ++residual_count;
  }
  if (residual_count < 2) return false;

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 100;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.function_tolerance = 1e-12;
  solver_options.gradient_tolerance = 1e-12;
  solver_options.parameter_tolerance = 1e-12;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  summary_text = summary.BriefReport();
  if (!summary.IsSolutionUsable()) return false;

  Eigen::Vector3d direction =
    line.direction + parameters[0] * basis_first + parameters[1] * basis_second;
  direction.normalize();
  Eigen::Vector3d origin = parameters[2] * basis_first + parameters[3] * basis_second;
  origin -= direction * direction.dot(origin);
  line = canonical_line(origin, direction);

  Eigen::Vector3d mean_point = Eigen::Vector3d::Zero();
  for (const size_t index : indices) mean_point += samples[index].point_camera;
  mean_point /= static_cast<double>(indices.size());
  if (line.direction.dot(mean_point - line.origin) < 0.0) line.direction *= -1.0;
  return line.origin.allFinite() && line.direction.allFinite();
}

struct SubsetFit
{
  LaserLine line;
  std::vector<size_t> inliers;
  std::string ceres_summary;
};

std::optional<SubsetFit> fit_subset(
  const std::vector<ProcessedSample> & samples, const std::vector<size_t> & indices,
  const CameraModel & camera, const FitOptions & options, uint32_t seed)
{
  if (indices.size() < 2) return std::nullopt;
  std::mt19937 generator(seed);
  std::uniform_int_distribution<size_t> distribution(0, indices.size() - 1);
  std::vector<size_t> best_inliers;
  double best_error_sum = std::numeric_limits<double>::infinity();

  for (int iteration = 0; iteration < options.ransac_iterations; ++iteration) {
    size_t first_position = distribution(generator);
    size_t second_position = distribution(generator);
    if (first_position == second_position) continue;
    const Eigen::Vector3d first = samples[indices[first_position]].point_camera;
    const Eigen::Vector3d second = samples[indices[second_position]].point_camera;
    if ((second - first).norm() < 1e-4) continue;
    const LaserLine candidate = canonical_line(first, second - first);

    std::vector<size_t> inliers;
    double error_sum = 0.0;
    for (const size_t index : indices) {
      const double error = point_line_distance(samples[index].point_camera, candidate);
      if (error <= options.ransac_threshold_m) {
        inliers.push_back(index);
        error_sum += error;
      }
    }
    if (
      inliers.size() > best_inliers.size() ||
      (inliers.size() == best_inliers.size() && error_sum < best_error_sum)) {
      best_inliers = std::move(inliers);
      best_error_sum = error_sum;
    }
  }
  if (best_inliers.size() < 2) return std::nullopt;

  auto pca = pca_line(samples, best_inliers);
  if (!pca) return std::nullopt;
  SubsetFit fit;
  fit.line = *pca;
  fit.inliers = best_inliers;
  if (!refine_line(samples, fit.inliers, camera, options, fit.line, fit.ceres_summary)) {
    return std::nullopt;
  }

  fit.inliers.clear();
  for (const size_t index : indices) {
    if (point_line_distance(samples[index].point_camera, fit.line) <= options.ransac_threshold_m) {
      fit.inliers.push_back(index);
    }
  }
  return fit;
}

ErrorStats pixel_stats(
  const LaserLine & line, const std::vector<ProcessedSample> & samples,
  const std::vector<size_t> & indices, const CameraModel & camera)
{
  std::vector<double> errors;
  for (const size_t index : indices) {
    const auto projected = project_line_to_board(line, samples[index].board, camera);
    if (projected) errors.push_back(cv::norm(*projected - samples[index].raw.laser_pixel));
  }
  return compute_stats(errors);
}

void bootstrap_uncertainty(
  const std::vector<ProcessedSample> & samples, const std::vector<size_t> & indices,
  const LaserLine & reference, const FitOptions & options, double & direction_p95_deg,
  double & origin_p95_m)
{
  std::mt19937 generator(options.random_seed ^ 0xB00757A9U);
  std::uniform_int_distribution<size_t> distribution(0, indices.size() - 1);
  std::vector<double> direction_errors;
  std::vector<double> origin_errors;
  for (int iteration = 0; iteration < options.bootstrap_iterations; ++iteration) {
    std::vector<size_t> resampled;
    resampled.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      resampled.push_back(indices[distribution(generator)]);
    }
    const auto line = pca_line(samples, resampled);
    if (!line) continue;
    direction_errors.push_back(line_direction_error_deg(*line, reference));
    origin_errors.push_back((line->origin - reference.origin).norm());
  }
  direction_p95_deg = percentile95(direction_errors);
  origin_p95_m = percentile95(origin_errors);
}

void append_quality_failure(FitResult & result, bool condition, const std::string & message)
{
  if (!condition) result.failure_reasons.push_back(message);
}

}  // namespace

std::vector<cv::Point3f> board_object_points(const BoardConfig & config)
{
  std::vector<cv::Point3f> points;
  points.reserve(config.pattern_cols * config.pattern_rows);
  for (int row = 0; row < config.pattern_rows; ++row) {
    for (int column = 0; column < config.pattern_cols; ++column) {
      points.emplace_back(
        static_cast<float>(column * config.square_size_m),
        static_cast<float>(row * config.square_size_m), 0.0F);
    }
  }
  return points;
}

BoardObservation observe_board(
  const cv::Mat & image, const CameraModel & camera, const BoardConfig & board)
{
  BoardObservation observation;
  if (image.empty()) {
    observation.rejection_reason = "empty_image";
    return observation;
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  const cv::Size pattern_size(board.pattern_cols, board.pattern_rows);
  if (!cv::findChessboardCornersSB(
        gray, pattern_size, observation.corners, cv::CALIB_CB_NORMALIZE_IMAGE)) {
    observation.rejection_reason = "board_not_found";
    return observation;
  }
  cv::cornerSubPix(
    gray, observation.corners, cv::Size(11, 11), cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001));

  const auto object_points = board_object_points(board);
  if (!cv::solvePnP(
        object_points, observation.corners, camera.camera_matrix, camera.distort_coeffs,
        observation.rvec, observation.tvec, false, cv::SOLVEPNP_ITERATIVE)) {
    observation.rejection_reason = "board_pnp_failed";
    return observation;
  }

  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(
    object_points, observation.rvec, observation.tvec, camera.camera_matrix, camera.distort_coeffs,
    reprojected);
  double squared_sum = 0.0;
  for (size_t i = 0; i < reprojected.size(); ++i) {
    const double error = cv::norm(reprojected[i] - observation.corners[i]);
    squared_sum += error * error;
  }
  observation.reprojection_rmse_px = std::sqrt(squared_sum / reprojected.size());

  cv::Mat rotation_cv;
  cv::Rodrigues(observation.rvec, rotation_cv);
  Eigen::Matrix3d rotation;
  cv::cv2eigen(rotation_cv, rotation);
  Eigen::Vector3d translation;
  cv::cv2eigen(observation.tvec, translation);
  observation.plane_normal_camera = (rotation * Eigen::Vector3d::UnitZ()).normalized();
  observation.plane_offset_camera = -observation.plane_normal_camera.dot(translation);
  observation.valid = true;
  return observation;
}

std::optional<Eigen::Vector3d> intersect_pixel_with_plane(
  const cv::Point2f & pixel, const CameraModel & camera, const Eigen::Vector3d & plane_normal,
  double plane_offset)
{
  std::vector<cv::Point2f> input{pixel};
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(input, normalized, camera.camera_matrix, camera.distort_coeffs);
  if (normalized.empty()) return std::nullopt;
  const Eigen::Vector3d ray(normalized[0].x, normalized[0].y, 1.0);
  const double denominator = plane_normal.dot(ray);
  if (std::abs(denominator) < 1e-8) return std::nullopt;
  const double lambda = -plane_offset / denominator;
  if (!std::isfinite(lambda) || lambda <= 0.0) return std::nullopt;
  const Eigen::Vector3d point = lambda * ray;
  if (!point.allFinite() || point.z() <= 0.0) return std::nullopt;
  return point;
}

ProcessedSample process_sample(
  const RawSample & sample, const cv::Mat & image, const CameraModel & camera,
  const BoardConfig & board, const FitOptions & options)
{
  ProcessedSample processed;
  processed.raw = sample;
  if (image.empty()) {
    processed.rejection_reason = "image_load_failed";
    return processed;
  }
  if (image.size() != sample.image_size) {
    processed.rejection_reason = "image_size_mismatch";
    return processed;
  }
  if (
    sample.laser_pixel.x < 0.0F || sample.laser_pixel.y < 0.0F ||
    sample.laser_pixel.x >= image.cols || sample.laser_pixel.y >= image.rows) {
    processed.rejection_reason = "laser_pixel_outside_image";
    return processed;
  }

  processed.board = observe_board(image, camera, board);
  if (!processed.board.valid) {
    processed.rejection_reason = processed.board.rejection_reason;
    return processed;
  }
  if (processed.board.reprojection_rmse_px > options.max_board_reprojection_rmse_px) {
    processed.rejection_reason = "board_reprojection_too_large";
    return processed;
  }

  const auto intersection = intersect_pixel_with_plane(
    sample.laser_pixel, camera, processed.board.plane_normal_camera,
    processed.board.plane_offset_camera);
  if (!intersection) {
    processed.rejection_reason = "laser_ray_plane_intersection_failed";
    return processed;
  }
  processed.point_camera = *intersection;
  processed.valid = true;
  return processed;
}

std::optional<cv::Point2f> project_line_to_board(
  const LaserLine & line, const BoardObservation & board, const CameraModel & camera,
  Eigen::Vector3d * intersection_camera)
{
  const double denominator = board.plane_normal_camera.dot(line.direction);
  if (std::abs(denominator) < 1e-8) return std::nullopt;
  const double lambda =
    -(board.plane_normal_camera.dot(line.origin) + board.plane_offset_camera) / denominator;
  if (!std::isfinite(lambda) || lambda <= 0.0) return std::nullopt;
  const Eigen::Vector3d point = line.origin + lambda * line.direction;
  if (!point.allFinite() || point.z() <= 0.0) return std::nullopt;
  if (intersection_camera != nullptr) *intersection_camera = point;

  const std::vector<cv::Point3f> points{cv::Point3f(
    static_cast<float>(point.x()), static_cast<float>(point.y()), static_cast<float>(point.z()))};
  std::vector<cv::Point2f> pixels;
  cv::projectPoints(
    points, cv::Vec3d::all(0), cv::Vec3d::all(0), camera.camera_matrix, camera.distort_coeffs,
    pixels);
  if (pixels.empty()) return std::nullopt;
  return pixels.front();
}

FitResult fit_laser_ray(
  std::vector<ProcessedSample> samples, const CameraModel & camera, const FitOptions & options)
{
  FitResult result;
  result.samples = std::move(samples);
  std::vector<size_t> valid_indices;
  for (size_t i = 0; i < result.samples.size(); ++i) {
    if (result.samples[i].valid) valid_indices.push_back(i);
  }
  result.valid_sample_count = static_cast<int>(valid_indices.size());
  if (valid_indices.size() < 2) {
    result.failure_reasons.push_back("fewer_than_two_valid_samples");
    return result;
  }

  std::vector<size_t> training_indices;
  std::vector<size_t> validation_indices;
  for (size_t ordinal = 0; ordinal < valid_indices.size(); ++ordinal) {
    const size_t index = valid_indices[ordinal];
    if (ordinal % 5 == 4) {
      validation_indices.push_back(index);
      result.samples[index].validation_sample = true;
    } else {
      training_indices.push_back(index);
    }
  }
  result.training_count = static_cast<int>(training_indices.size());
  result.validation_count = static_cast<int>(validation_indices.size());

  const auto training_fit =
    fit_subset(result.samples, training_indices, camera, options, options.random_seed);
  if (!training_fit) {
    result.failure_reasons.push_back("training_fit_failed");
    return result;
  }
  result.training_pixel_error_px =
    pixel_stats(training_fit->line, result.samples, training_fit->inliers, camera);
  result.validation_pixel_error_px =
    pixel_stats(training_fit->line, result.samples, validation_indices, camera);

  const auto final_fit =
    fit_subset(result.samples, valid_indices, camera, options, options.random_seed ^ 0xF17A11U);
  if (!final_fit) {
    result.failure_reasons.push_back("final_fit_failed");
    return result;
  }
  result.fit_succeeded = true;
  result.line = final_fit->line;
  result.ceres_summary = final_fit->ceres_summary;
  result.inlier_count = static_cast<int>(final_fit->inliers.size());
  result.inlier_ratio = static_cast<double>(result.inlier_count) / valid_indices.size();

  const std::set<size_t> final_inliers(final_fit->inliers.begin(), final_fit->inliers.end());
  std::vector<double> point_errors;
  std::vector<double> projections;
  for (const size_t index : valid_indices) {
    auto & sample = result.samples[index];
    sample.ransac_inlier = final_inliers.count(index) != 0;
    sample.point_line_error_m = point_line_distance(sample.point_camera, result.line);
    const auto pixel = project_line_to_board(result.line, sample.board, camera);
    if (pixel) {
      sample.predicted_pixel = *pixel;
      sample.pixel_error_px = cv::norm(*pixel - sample.raw.laser_pixel);
    } else {
      sample.pixel_error_px = nan_value();
    }
    if (sample.ransac_inlier) {
      point_errors.push_back(sample.point_line_error_m);
      projections.push_back(result.line.direction.dot(sample.point_camera - result.line.origin));
    }
  }
  result.point_line_error_m = compute_stats(point_errors);

  if (!projections.empty()) {
    const auto [minimum, maximum] = std::minmax_element(projections.begin(), projections.end());
    result.sample_span_m = *maximum - *minimum;
    if (result.sample_span_m > 0.0) {
      for (const double projection : projections) {
        const double normalized = (projection - *minimum) / result.sample_span_m;
        const int bin = std::min(2, static_cast<int>(normalized * 3.0));
        ++result.distance_bin_counts[bin];
      }
    }
  }

  bootstrap_uncertainty(
    result.samples, final_fit->inliers, result.line, options, result.bootstrap_direction_p95_deg,
    result.bootstrap_origin_p95_m);

  append_quality_failure(
    result, result.valid_sample_count >= options.min_valid_samples, "insufficient_valid_samples");
  append_quality_failure(result, result.validation_count >= 3, "insufficient_validation_samples");
  append_quality_failure(
    result,
    result.distance_bin_counts[0] >= options.min_samples_per_distance_bin &&
      result.distance_bin_counts[1] >= options.min_samples_per_distance_bin &&
      result.distance_bin_counts[2] >= options.min_samples_per_distance_bin,
    "insufficient_distance_bin_coverage");
  append_quality_failure(
    result, result.sample_span_m >= options.min_sample_span_m, "insufficient_sample_span");
  append_quality_failure(
    result, result.inlier_ratio >= options.min_inlier_ratio, "ransac_inlier_ratio_too_low");
  append_quality_failure(
    result,
    std::isfinite(result.point_line_error_m.rms) &&
      result.point_line_error_m.rms <= options.max_point_line_rms_m,
    "point_line_rms_too_large");
  append_quality_failure(
    result,
    std::isfinite(result.validation_pixel_error_px.rms) &&
      result.validation_pixel_error_px.rms <= options.max_validation_rms_px,
    "validation_pixel_rms_too_large");
  append_quality_failure(
    result,
    std::isfinite(result.validation_pixel_error_px.p95) &&
      result.validation_pixel_error_px.p95 <= options.max_validation_p95_px,
    "validation_pixel_p95_too_large");
  append_quality_failure(
    result,
    std::isfinite(result.bootstrap_direction_p95_deg) &&
      result.bootstrap_direction_p95_deg <= options.max_bootstrap_direction_p95_deg,
    "bootstrap_direction_uncertainty_too_large");
  result.quality_passed = result.failure_reasons.empty();
  return result;
}

double point_line_distance(const Eigen::Vector3d & point, const LaserLine & line)
{
  return (point - line.origin).cross(line.direction.normalized()).norm();
}

double line_direction_error_deg(const LaserLine & lhs, const LaserLine & rhs)
{
  const double cosine =
    std::clamp(std::abs(lhs.direction.normalized().dot(rhs.direction.normalized())), 0.0, 1.0);
  return std::acos(cosine) * kRadToDeg;
}

}  // namespace laser_calibration
