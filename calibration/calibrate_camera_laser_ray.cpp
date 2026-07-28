#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "calibration/laser_ray_calibration.hpp"
#include "io/camera.hpp"
#include "tools/logger.hpp"

namespace fs = std::filesystem;
using laser_calibration::BoardConfig;
using laser_calibration::CameraModel;
using laser_calibration::FitOptions;
using laser_calibration::FitResult;
using laser_calibration::LaserLine;
using laser_calibration::ProcessedSample;
using laser_calibration::RawSample;

namespace
{

constexpr char kWindowName[] = "Camera Laser Ray Calibration";
constexpr char kOfflineWindowName[] = "Offline Laser Point Annotation";

struct CalibrationConfig
{
  CameraModel camera;
  BoardConfig board;
  Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();
  std::optional<cv::Size> calibrated_image_size;
};

struct MouseState
{
  double scale = 1.0;
  int display_width = 0;
  int display_height = 0;
  std::optional<cv::Point2f> clicked_pixel;
};

void mouse_callback(int event, int x, int y, int, void * userdata)
{
  if (event != cv::EVENT_LBUTTONDOWN) return;
  auto * state = static_cast<MouseState *>(userdata);
  if (
    state->scale <= 0.0 || x < 0 || y < 0 || x >= state->display_width ||
    y >= state->display_height) {
    return;
  }
  state->clicked_pixel =
    cv::Point2f(static_cast<float>(x / state->scale), static_cast<float>(y / state->scale));
}

std::vector<double> node_vector(const YAML::Node & yaml, const std::string & key, size_t minimum)
{
  if (!yaml[key]) throw std::runtime_error("Missing YAML key: " + key);
  auto values = yaml[key].as<std::vector<double>>();
  if (values.size() < minimum) throw std::runtime_error("Invalid YAML vector: " + key);
  return values;
}

CalibrationConfig load_config(const std::string & path)
{
  const auto yaml = YAML::LoadFile(path);
  CalibrationConfig config;
  config.board.pattern_cols = yaml["pattern_cols"].as<int>();
  config.board.pattern_rows = yaml["pattern_rows"].as<int>();
  config.board.square_size_m = yaml["square_size_mm"].as<double>() * 1e-3;
  if (
    config.board.pattern_cols <= 0 || config.board.pattern_rows <= 0 ||
    config.board.square_size_m <= 0.0) {
    throw std::runtime_error("Invalid chessboard configuration");
  }

  const auto camera_values = node_vector(yaml, "camera_matrix", 9);
  if (camera_values.size() != 9) throw std::runtime_error("camera_matrix must contain 9 values");
  config.camera.camera_matrix = cv::Mat(3, 3, CV_64F);
  std::copy(
    camera_values.begin(), camera_values.begin() + 9, config.camera.camera_matrix.ptr<double>());
  const auto distortion_values = node_vector(yaml, "distort_coeffs", 4);
  if (distortion_values.size() != 4 && distortion_values.size() != 5) {
    throw std::runtime_error("distort_coeffs must use [k1,k2,p1,p2] or [k1,k2,p1,p2,k3]");
  }
  config.camera.distort_coeffs = cv::Mat(1, static_cast<int>(distortion_values.size()), CV_64F);
  std::copy(
    distortion_values.begin(), distortion_values.end(), config.camera.distort_coeffs.ptr<double>());

  const auto rotation_values = node_vector(yaml, "R_camera2gimbal", 9);
  const auto translation_values = node_vector(yaml, "t_camera2gimbal", 3);
  if (rotation_values.size() != 9 || translation_values.size() != 3) {
    throw std::runtime_error("R_camera2gimbal/t_camera2gimbal must contain 9/3 values");
  }
  config.R_camera2gimbal =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation_values.data());
  config.t_camera2gimbal = Eigen::Map<const Eigen::Vector3d>(translation_values.data());
  if (yaml["image_width"] || yaml["image_height"]) {
    if (!yaml["image_width"] || !yaml["image_height"]) {
      throw std::runtime_error("image_width and image_height must be provided together");
    }
    const cv::Size image_size(yaml["image_width"].as<int>(), yaml["image_height"].as<int>());
    if (image_size.width <= 0 || image_size.height <= 0) {
      throw std::runtime_error("Invalid calibrated image size");
    }
    config.calibrated_image_size = image_size;
  }
  return config;
}

std::vector<double> mat_values(const cv::Mat & matrix)
{
  const cv::Mat flattened = matrix.reshape(1, 1);
  return std::vector<double>(flattened.ptr<double>(), flattened.ptr<double>() + flattened.total());
}

std::vector<double> eigen_values(const Eigen::Vector3d & value)
{
  return {value.x(), value.y(), value.z()};
}

std::string current_time_string()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&time, &local_time);
  std::ostringstream output;
  output << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S%z");
  return output.str();
}

fs::path samples_path(const fs::path & dataset) { return dataset / "samples.yaml"; }

std::vector<RawSample> load_raw_samples(const fs::path & dataset)
{
  std::vector<RawSample> samples;
  const fs::path path = samples_path(dataset);
  if (!fs::exists(path)) return samples;
  const auto yaml = YAML::LoadFile(path.string());
  if (!yaml["samples"]) return samples;
  for (const auto & node : yaml["samples"]) {
    RawSample sample;
    sample.id = node["id"].as<int>();
    sample.image_path = node["image"].as<std::string>();
    const auto pixel = node["laser_pixel"].as<std::vector<float>>();
    if (pixel.size() != 2) throw std::runtime_error("laser_pixel must contain two values");
    sample.laser_pixel = {pixel[0], pixel[1]};
    sample.image_size = {node["image_width"].as<int>(), node["image_height"].as<int>()};
    sample.timestamp_ns = node["timestamp_ns"].as<int64_t>(0);
    samples.push_back(sample);
  }
  std::sort(samples.begin(), samples.end(), [](const auto & lhs, const auto & rhs) {
    return lhs.id < rhs.id;
  });
  return samples;
}

void save_raw_samples(const fs::path & dataset, const std::vector<RawSample> & samples)
{
  fs::create_directories(dataset);
  YAML::Emitter output;
  output << YAML::BeginMap;
  output << YAML::Key << "version" << YAML::Value << 1;
  output << YAML::Key << "samples" << YAML::Value << YAML::BeginSeq;
  for (const auto & sample : samples) {
    output << YAML::BeginMap;
    output << YAML::Key << "id" << YAML::Value << sample.id;
    output << YAML::Key << "image" << YAML::Value << sample.image_path;
    output << YAML::Key << "laser_pixel" << YAML::Value << YAML::Flow
           << std::vector<float>{sample.laser_pixel.x, sample.laser_pixel.y};
    output << YAML::Key << "image_width" << YAML::Value << sample.image_size.width;
    output << YAML::Key << "image_height" << YAML::Value << sample.image_size.height;
    output << YAML::Key << "timestamp_ns" << YAML::Value << sample.timestamp_ns;
    output << YAML::EndMap;
  }
  output << YAML::EndSeq << YAML::EndMap;
  if (!output.good()) throw std::runtime_error("Failed to serialize samples.yaml");

  const fs::path temporary = samples_path(dataset).string() + ".tmp";
  {
    std::ofstream stream(temporary);
    if (!stream) throw std::runtime_error("Failed to write samples.yaml");
    stream << output.c_str() << '\n';
  }
  fs::rename(temporary, samples_path(dataset));
}

int next_sample_id(const std::vector<RawSample> & samples)
{
  return samples.empty() ? 1 : samples.back().id + 1;
}

bool is_supported_image(const fs::path & path)
{
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
         extension == ".bmp" || extension == ".tif" || extension == ".tiff";
}

std::optional<unsigned long long> numeric_stem(const fs::path & path)
{
  const std::string stem = path.stem().string();
  if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](unsigned char value) {
        return std::isdigit(value);
      })) {
    return std::nullopt;
  }
  try {
    return std::stoull(stem);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

std::vector<fs::path> discover_dataset_images(const fs::path & dataset)
{
  std::vector<fs::path> images;
  std::set<std::string> unique_paths;
  const std::vector<fs::path> search_folders{dataset, dataset / "images"};
  for (const auto & folder : search_folders) {
    if (!fs::is_directory(folder)) continue;
    for (const auto & entry : fs::directory_iterator(folder)) {
      if (!entry.is_regular_file() || !is_supported_image(entry.path())) continue;
      const fs::path relative = entry.path().lexically_relative(dataset);
      const std::string key = relative.generic_string();
      if (unique_paths.insert(key).second) images.push_back(relative);
    }
  }
  std::sort(images.begin(), images.end(), [](const fs::path & lhs, const fs::path & rhs) {
    const auto lhs_number = numeric_stem(lhs);
    const auto rhs_number = numeric_stem(rhs);
    if (lhs_number && rhs_number && *lhs_number != *rhs_number) return *lhs_number < *rhs_number;
    if (lhs_number.has_value() != rhs_number.has_value()) return lhs_number.has_value();
    return lhs.generic_string() < rhs.generic_string();
  });
  return images;
}

void draw_label(
  cv::Mat & image, const std::string & text, const cv::Point & position, const cv::Scalar & color)
{
  cv::putText(
    image, text, position, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv::LINE_AA);
}

cv::Mat fit_for_display(const cv::Mat & image, double & scale)
{
  scale = std::min(
    {1.0, 1400.0 / static_cast<double>(image.cols), 850.0 / static_cast<double>(image.rows)});
  if (scale >= 1.0) return image.clone();
  cv::Mat resized;
  cv::resize(image, resized, {}, scale, scale, cv::INTER_AREA);
  return resized;
}

bool annotate_dataset_images(
  const fs::path & dataset, const CalibrationConfig & config, const FitOptions & options)
{
  const auto image_paths = discover_dataset_images(dataset);
  if (image_paths.empty()) {
    tools::logger()->warn("[LaserCalibration] No images found in {}", dataset.string());
    return true;
  }

  std::vector<RawSample> samples;
  MouseState mouse;
  cv::namedWindow(kOfflineWindowName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(kOfflineWindowName, mouse_callback, &mouse);
  for (size_t image_index = 0; image_index < image_paths.size(); ++image_index) {
    const fs::path & relative_path = image_paths[image_index];
    const cv::Mat raw_image = cv::imread((dataset / relative_path).string());
    if (raw_image.empty()) {
      tools::logger()->warn(
        "[LaserCalibration] Failed to load {}; skipped", relative_path.string());
      continue;
    }
    const auto board = laser_calibration::observe_board(raw_image, config.camera, config.board);

    bool advance = false;
    while (!advance) {
      cv::Mat drawing = raw_image.clone();
      if (!board.corners.empty()) {
        cv::drawChessboardCorners(
          drawing, cv::Size(config.board.pattern_cols, config.board.pattern_rows), board.corners,
          board.valid);
      }
      if (mouse.clicked_pixel) {
        cv::drawMarker(
          drawing, *mouse.clicked_pixel, {0, 0, 255}, cv::MARKER_CROSS, 26, 3, cv::LINE_AA);
      }

      const bool clicked_inside =
        mouse.clicked_pixel && mouse.clicked_pixel->x >= 0.0F && mouse.clicked_pixel->y >= 0.0F &&
        mouse.clicked_pixel->x < raw_image.cols && mouse.clicked_pixel->y < raw_image.rows;
      const bool image_size_valid =
        !config.calibrated_image_size || raw_image.size() == *config.calibrated_image_size;
      bool sample_ready = board.valid && clicked_inside && image_size_valid &&
                          board.reprojection_rmse_px <= options.max_board_reprojection_rmse_px;
      if (sample_ready) {
        sample_ready = laser_calibration::intersect_pixel_with_plane(
                         *mouse.clicked_pixel, config.camera, board.plane_normal_camera,
                         board.plane_offset_camera)
                         .has_value();
      }

      draw_label(
        drawing,
        fmt::format(
          "image {}/{}: {}", image_index + 1, image_paths.size(), relative_path.generic_string()),
        {24, 38}, {230, 230, 230});
      draw_label(
        drawing,
        fmt::format(
          "board={} rmse={:.3f}px saved={}", board.valid, board.reprojection_rmse_px,
          samples.size()),
        {24, 72}, board.valid ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 80, 255));
      draw_label(
        drawing,
        sample_ready ? "READY: s save | r clear | k skip | q abort"
                     : "click laser spot | r clear | k skip | q abort",
        {24, 106}, sample_ready ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255));

      double scale = 1.0;
      cv::Mat display = fit_for_display(drawing, scale);
      mouse.scale = scale;
      mouse.display_width = display.cols;
      mouse.display_height = display.rows;
      cv::imshow(kOfflineWindowName, display);
      const int key = cv::waitKey(20) & 0xFF;
      if (key == 'q') {
        cv::destroyWindow(kOfflineWindowName);
        tools::logger()->warn("[LaserCalibration] Offline annotation aborted");
        return false;
      }
      if (key == 'r') mouse.clicked_pixel.reset();
      if (key == 'k') {
        tools::logger()->info(
          "[LaserCalibration] Skipped image {}", relative_path.generic_string());
        advance = true;
      }
      if (key == 's') {
        if (!sample_ready) {
          tools::logger()->warn("[LaserCalibration] Current image/click is not valid");
          continue;
        }
        RawSample sample;
        sample.id = next_sample_id(samples);
        sample.image_path = relative_path.generic_string();
        sample.laser_pixel = *mouse.clicked_pixel;
        sample.image_size = raw_image.size();
        sample.timestamp_ns = 0;
        samples.push_back(sample);
        save_raw_samples(dataset, samples);
        tools::logger()->info(
          "[LaserCalibration] Annotated {} at ({:.2f}, {:.2f})", relative_path.generic_string(),
          sample.laser_pixel.x, sample.laser_pixel.y);
        advance = true;
      }
    }
  }
  cv::destroyWindow(kOfflineWindowName);
  return true;
}

void emit_stats(
  YAML::Emitter & output, const std::string & key, const laser_calibration::ErrorStats & stats)
{
  output << YAML::Key << key << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "count" << YAML::Value << stats.count;
  output << YAML::Key << "rms" << YAML::Value << stats.rms;
  output << YAML::Key << "p95" << YAML::Value << stats.p95;
  output << YAML::Key << "max" << YAML::Value << stats.maximum;
  output << YAML::EndMap;
}

void write_sample_report(const fs::path & dataset, const FitResult & result)
{
  std::ofstream output(dataset / "sample_report.csv");
  if (!output) throw std::runtime_error("Failed to write sample_report.csv");
  output << "id,image,valid,rejection,validation,ransac_inlier,board_rmse_px,point_x_m,"
            "point_y_m,point_z_m,point_line_error_m,clicked_u,clicked_v,predicted_u,"
            "predicted_v,pixel_error_px\n";
  output << std::setprecision(10);
  for (const auto & sample : result.samples) {
    output << sample.raw.id << ',' << sample.raw.image_path << ',' << sample.valid << ','
           << sample.rejection_reason << ',' << sample.validation_sample << ','
           << sample.ransac_inlier << ',' << sample.board.reprojection_rmse_px << ','
           << sample.point_camera.x() << ',' << sample.point_camera.y() << ','
           << sample.point_camera.z() << ',' << sample.point_line_error_m << ','
           << sample.raw.laser_pixel.x << ',' << sample.raw.laser_pixel.y << ','
           << sample.predicted_pixel.x << ',' << sample.predicted_pixel.y << ','
           << sample.pixel_error_px << '\n';
  }
}

void write_report(
  const fs::path & dataset, const FitResult & result, const CalibrationConfig & config,
  const FitOptions & options, const std::string & config_path)
{
  YAML::Emitter output;
  output << YAML::BeginMap;
  output << YAML::Key << "generated_at" << YAML::Value << current_time_string();
  output << YAML::Key << "source_config" << YAML::Value << config_path;
  output << YAML::Key << "fit_succeeded" << YAML::Value << result.fit_succeeded;
  output << YAML::Key << "quality_passed" << YAML::Value << result.quality_passed;
  output << YAML::Key << "failure_reasons" << YAML::Value << YAML::Flow << result.failure_reasons;
  output << YAML::Key << "sample_counts" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "total" << YAML::Value << result.samples.size();
  output << YAML::Key << "valid" << YAML::Value << result.valid_sample_count;
  output << YAML::Key << "training" << YAML::Value << result.training_count;
  output << YAML::Key << "validation" << YAML::Value << result.validation_count;
  output << YAML::Key << "inliers" << YAML::Value << result.inlier_count;
  output << YAML::Key << "distance_bins" << YAML::Value << YAML::Flow
         << std::vector<int>{
              result.distance_bin_counts[0], result.distance_bin_counts[1],
              result.distance_bin_counts[2]};
  output << YAML::EndMap;
  std::map<std::string, int> rejection_counts;
  std::map<std::pair<int, int>, int> image_size_counts;
  for (const auto & sample : result.samples) {
    if (!sample.valid) ++rejection_counts[sample.rejection_reason];
    ++image_size_counts[{sample.raw.image_size.width, sample.raw.image_size.height}];
  }
  output << YAML::Key << "rejection_counts" << YAML::Value << YAML::BeginMap;
  for (const auto & [reason, count] : rejection_counts) {
    output << YAML::Key << reason << YAML::Value << count;
  }
  output << YAML::EndMap;
  output << YAML::Key << "dataset_image_sizes" << YAML::Value << YAML::BeginSeq;
  for (const auto & [size, count] : image_size_counts) {
    output << YAML::BeginMap;
    output << YAML::Key << "width" << YAML::Value << size.first;
    output << YAML::Key << "height" << YAML::Value << size.second;
    output << YAML::Key << "samples" << YAML::Value << count;
    output << YAML::EndMap;
  }
  output << YAML::EndSeq;
  output << YAML::Key << "inlier_ratio" << YAML::Value << result.inlier_ratio;
  output << YAML::Key << "sample_span_m" << YAML::Value << result.sample_span_m;
  emit_stats(output, "point_line_error_m", result.point_line_error_m);
  emit_stats(output, "training_pixel_error_px", result.training_pixel_error_px);
  emit_stats(output, "validation_pixel_error_px", result.validation_pixel_error_px);
  output << YAML::Key << "bootstrap_direction_p95_deg" << YAML::Value
         << result.bootstrap_direction_p95_deg;
  output << YAML::Key << "bootstrap_origin_p95_m" << YAML::Value << result.bootstrap_origin_p95_m;
  output << YAML::Key << "ceres_summary" << YAML::Value << result.ceres_summary;
  if (result.fit_succeeded) {
    output << YAML::Key << "candidate_line" << YAML::Value << YAML::BeginMap;
    output << YAML::Key << "origin_in_camera_m" << YAML::Value << YAML::Flow
           << eigen_values(result.line.origin);
    output << YAML::Key << "direction_in_camera" << YAML::Value << YAML::Flow
           << eigen_values(result.line.direction);
    output << YAML::EndMap;
  }

  output << YAML::Key << "camera" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "camera_matrix" << YAML::Value << YAML::Flow
         << mat_values(config.camera.camera_matrix);
  output << YAML::Key << "distort_coeffs" << YAML::Value << YAML::Flow
         << mat_values(config.camera.distort_coeffs);
  if (config.calibrated_image_size) {
    output << YAML::Key << "calibrated_image_width" << YAML::Value
           << config.calibrated_image_size->width;
    output << YAML::Key << "calibrated_image_height" << YAML::Value
           << config.calibrated_image_size->height;
  } else {
    output << YAML::Key << "calibrated_image_size_available" << YAML::Value << false;
  }
  output << YAML::EndMap;
  output << YAML::Key << "board" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "pattern_cols" << YAML::Value << config.board.pattern_cols;
  output << YAML::Key << "pattern_rows" << YAML::Value << config.board.pattern_rows;
  output << YAML::Key << "square_size_m" << YAML::Value << config.board.square_size_m;
  output << YAML::EndMap;
  output << YAML::Key << "thresholds" << YAML::Value << YAML::BeginMap;
  output << YAML::Key << "max_board_reprojection_rmse_px" << YAML::Value
         << options.max_board_reprojection_rmse_px;
  output << YAML::Key << "ransac_threshold_m" << YAML::Value << options.ransac_threshold_m;
  output << YAML::Key << "ransac_iterations" << YAML::Value << options.ransac_iterations;
  output << YAML::Key << "bootstrap_iterations" << YAML::Value << options.bootstrap_iterations;
  output << YAML::Key << "huber_loss_px" << YAML::Value << options.huber_loss_px;
  output << YAML::Key << "random_seed" << YAML::Value << options.random_seed;
  output << YAML::Key << "min_valid_samples" << YAML::Value << options.min_valid_samples;
  output << YAML::Key << "min_samples_per_distance_bin" << YAML::Value
         << options.min_samples_per_distance_bin;
  output << YAML::Key << "min_sample_span_m" << YAML::Value << options.min_sample_span_m;
  output << YAML::Key << "min_inlier_ratio" << YAML::Value << options.min_inlier_ratio;
  output << YAML::Key << "max_point_line_rms_m" << YAML::Value << options.max_point_line_rms_m;
  output << YAML::Key << "max_validation_rms_px" << YAML::Value << options.max_validation_rms_px;
  output << YAML::Key << "max_validation_p95_px" << YAML::Value << options.max_validation_p95_px;
  output << YAML::Key << "max_bootstrap_direction_p95_deg" << YAML::Value
         << options.max_bootstrap_direction_p95_deg;
  output << YAML::EndMap;
  output << YAML::Key << "samples" << YAML::Value << YAML::BeginSeq;
  for (const auto & sample : result.samples) {
    output << YAML::BeginMap;
    output << YAML::Key << "id" << YAML::Value << sample.raw.id;
    output << YAML::Key << "image" << YAML::Value << sample.raw.image_path;
    output << YAML::Key << "valid" << YAML::Value << sample.valid;
    output << YAML::Key << "rejection" << YAML::Value << sample.rejection_reason;
    output << YAML::Key << "validation" << YAML::Value << sample.validation_sample;
    output << YAML::Key << "ransac_inlier" << YAML::Value << sample.ransac_inlier;
    output << YAML::Key << "board_rmse_px" << YAML::Value << sample.board.reprojection_rmse_px;
    output << YAML::Key << "point_camera_m" << YAML::Value << YAML::Flow
           << eigen_values(sample.point_camera);
    output << YAML::Key << "point_line_error_m" << YAML::Value << sample.point_line_error_m;
    output << YAML::Key << "pixel_error_px" << YAML::Value << sample.pixel_error_px;
    output << YAML::EndMap;
  }
  output << YAML::EndSeq << YAML::EndMap;

  std::ofstream stream(dataset / "laser_ray_report.yaml");
  if (!stream) throw std::runtime_error("Failed to write laser_ray_report.yaml");
  stream << output.c_str() << '\n';
}

LaserLine transform_line_to_gimbal(const LaserLine & camera_line, const CalibrationConfig & config)
{
  LaserLine line;
  line.direction = (config.R_camera2gimbal * camera_line.direction).normalized();
  const Eigen::Vector3d point =
    config.R_camera2gimbal * camera_line.origin + config.t_camera2gimbal;
  line.origin = point - line.direction * line.direction.dot(point);
  return line;
}

void write_ray_yaml(
  const fs::path & dataset, const FitResult & result, const CalibrationConfig & config)
{
  const LaserLine gimbal_line = transform_line_to_gimbal(result.line, config);
  YAML::Emitter output;
  output << YAML::BeginMap;
  output << YAML::Key << "generated_at" << YAML::Value << current_time_string();
  output << YAML::Key << "quality" << YAML::Value << "passed";
  output << YAML::Key << "valid_samples" << YAML::Value << result.valid_sample_count;
  output << YAML::Key << "inlier_samples" << YAML::Value << result.inlier_count;
  output << YAML::Key << "inlier_ratio" << YAML::Value << result.inlier_ratio;
  output << YAML::Key << "point_line_rms_m" << YAML::Value << result.point_line_error_m.rms;
  output << YAML::Key << "validation_pixel_rms_px" << YAML::Value
         << result.validation_pixel_error_px.rms;
  output << YAML::Key << "validation_pixel_p95_px" << YAML::Value
         << result.validation_pixel_error_px.p95;
  output << YAML::Key << "bootstrap_direction_p95_deg" << YAML::Value
         << result.bootstrap_direction_p95_deg;
  output << YAML::Key << "bootstrap_origin_p95_m" << YAML::Value << result.bootstrap_origin_p95_m;
  output << YAML::Key << "laser_line_origin_in_camera_m" << YAML::Value << YAML::Flow
         << eigen_values(result.line.origin);
  output << YAML::Key << "laser_line_direction_in_camera" << YAML::Value << YAML::Flow
         << eigen_values(result.line.direction);
  output << YAML::Key << "laser_line_origin_in_gimbal_m" << YAML::Value << YAML::Flow
         << eigen_values(gimbal_line.origin);
  output << YAML::Key << "laser_line_direction_in_gimbal" << YAML::Value << YAML::Flow
         << eigen_values(gimbal_line.direction);
  output << YAML::EndMap;
  std::ofstream stream(dataset / "laser_ray.yaml");
  if (!stream) throw std::runtime_error("Failed to write laser_ray.yaml");
  stream << output.c_str() << '\n';
}

void write_review_images(
  const fs::path & dataset, const FitResult & result, const CalibrationConfig & config)
{
  const fs::path review_folder = dataset / "review";
  fs::remove_all(review_folder);
  fs::create_directories(review_folder);
  for (const auto & sample : result.samples) {
    cv::Mat image = cv::imread((dataset / sample.raw.image_path).string());
    if (image.empty()) continue;
    if (!sample.board.corners.empty()) {
      cv::drawChessboardCorners(
        image, cv::Size(config.board.pattern_cols, config.board.pattern_rows), sample.board.corners,
        sample.board.valid);
    }
    cv::drawMarker(
      image, sample.raw.laser_pixel, {0, 0, 255}, cv::MARKER_CROSS, 24, 3, cv::LINE_AA);
    if (result.fit_succeeded && sample.valid && std::isfinite(sample.pixel_error_px)) {
      cv::drawMarker(
        image, sample.predicted_pixel, {0, 255, 0}, cv::MARKER_CROSS, 20, 2, cv::LINE_AA);
      cv::line(
        image, sample.raw.laser_pixel, sample.predicted_pixel, {0, 255, 255}, 2, cv::LINE_AA);
    }
    draw_label(
      image,
      fmt::format(
        "id={} valid={} inlier={} board={:.2f}px laser={:.2f}px", sample.raw.id, sample.valid,
        sample.ransac_inlier, sample.board.reprojection_rmse_px, sample.pixel_error_px),
      {24, 38}, sample.ransac_inlier ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 80, 255));
    cv::imwrite((review_folder / fmt::format("{:04d}.jpg", sample.raw.id)).string(), image);
  }
}

int fit_dataset(
  const fs::path & dataset, const CalibrationConfig & config, const FitOptions & options,
  const std::string & config_path, bool annotate_if_missing)
{
  auto raw_samples = load_raw_samples(dataset);
  if (raw_samples.empty() && annotate_if_missing) {
    if (!annotate_dataset_images(dataset, config, options)) return 0;
    raw_samples = load_raw_samples(dataset);
  }
  std::vector<ProcessedSample> processed;
  processed.reserve(raw_samples.size());
  std::optional<cv::Size> expected_size;
  for (const auto & sample : raw_samples) {
    cv::Mat image = cv::imread((dataset / sample.image_path).string());
    if (!expected_size && !image.empty()) expected_size = image.size();
    auto current =
      laser_calibration::process_sample(sample, image, config.camera, config.board, options);
    if (config.calibrated_image_size && sample.image_size != *config.calibrated_image_size) {
      current.valid = false;
      current.rejection_reason = "calibrated_image_size_mismatch";
    }
    if (expected_size && sample.image_size != *expected_size) {
      current.valid = false;
      current.rejection_reason = "dataset_image_size_mismatch";
    }
    processed.push_back(std::move(current));
  }

  FitResult result = laser_calibration::fit_laser_ray(processed, config.camera, options);
  fs::create_directories(dataset);
  write_report(dataset, result, config, options, config_path);
  write_sample_report(dataset, result);
  write_review_images(dataset, result, config);
  const fs::path ray_path = dataset / "laser_ray.yaml";
  if (result.quality_passed) {
    write_ray_yaml(dataset, result, config);
    tools::logger()->info(
      "[LaserCalibration] PASSED: {} valid, {:.1f}% inliers, validation RMS {:.3f}px",
      result.valid_sample_count, result.inlier_ratio * 100.0, result.validation_pixel_error_px.rms);
    return 0;
  }

  if (fs::exists(ray_path)) fs::remove(ray_path);
  tools::logger()->error(
    "[LaserCalibration] FAILED quality gate. Report: {}",
    (dataset / "laser_ray_report.yaml").string());
  for (const auto & reason : result.failure_reasons) {
    tools::logger()->error("  - {}", reason);
  }
  return 2;
}

FitOptions options_from_cli(const cv::CommandLineParser & cli)
{
  FitOptions options;
  options.max_board_reprojection_rmse_px = cli.get<double>("max-board-rmse-px");
  options.ransac_threshold_m = cli.get<double>("ransac-threshold-mm") * 1e-3;
  options.ransac_iterations = cli.get<int>("ransac-iterations");
  options.bootstrap_iterations = cli.get<int>("bootstrap-iterations");
  options.min_valid_samples = cli.get<int>("min-samples");
  options.min_samples_per_distance_bin = cli.get<int>("min-bin-samples");
  options.min_sample_span_m = cli.get<double>("min-span-m");
  options.min_inlier_ratio = cli.get<double>("min-inlier-ratio");
  options.max_point_line_rms_m = cli.get<double>("max-line-rms-mm") * 1e-3;
  options.max_validation_rms_px = cli.get<double>("max-validation-rms-px");
  options.max_validation_p95_px = cli.get<double>("max-validation-p95-px");
  options.max_bootstrap_direction_p95_deg = cli.get<double>("max-bootstrap-direction-deg");
  return options;
}

int capture_dataset(
  const fs::path & dataset, const std::string & config_path, const CalibrationConfig & config,
  const FitOptions & options, bool allow_fit)
{
  fs::create_directories(dataset / "images");
  auto samples = load_raw_samples(dataset);
  io::Camera camera(config_path);
  MouseState mouse;
  cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(kWindowName, mouse_callback, &mouse);

  cv::Mat waiting_image(180, 720, CV_8UC3, cv::Scalar(32, 32, 32));
  draw_label(waiting_image, "Waiting for camera frames...", {24, 54}, {230, 230, 230});
  draw_label(
    waiting_image, "Check camera connection/SN, or press q to quit", {24, 96},
    {0, 180, 255});
  cv::imshow(kWindowName, waiting_image);

  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  while (true) {
    if (!camera.try_read(image, timestamp)) {
      const int key = cv::waitKey(20) & 0xFF;
      if (key == 'q') break;
      continue;
    }
    if (image.empty()) continue;
    const cv::Mat raw_image = image.clone();
    const auto board = laser_calibration::observe_board(image, config.camera, config.board);
    if (!board.corners.empty()) {
      cv::drawChessboardCorners(
        image, cv::Size(config.board.pattern_cols, config.board.pattern_rows), board.corners,
        board.valid);
    }
    if (mouse.clicked_pixel) {
      cv::drawMarker(
        image, *mouse.clicked_pixel, {0, 0, 255}, cv::MARKER_CROSS, 26, 3, cv::LINE_AA);
    }

    const bool clicked_inside =
      mouse.clicked_pixel && mouse.clicked_pixel->x >= 0.0F && mouse.clicked_pixel->y >= 0.0F &&
      mouse.clicked_pixel->x < image.cols && mouse.clicked_pixel->y < image.rows;
    bool sample_ready = board.valid && clicked_inside &&
                        board.reprojection_rmse_px <= options.max_board_reprojection_rmse_px;
    std::optional<Eigen::Vector3d> point;
    if (sample_ready) {
      point = laser_calibration::intersect_pixel_with_plane(
        *mouse.clicked_pixel, config.camera, board.plane_normal_camera, board.plane_offset_camera);
      sample_ready = point.has_value();
    }

    draw_label(
      image,
      fmt::format(
        "samples={} board={} rmse={:.3f}px", samples.size(), board.valid,
        board.reprojection_rmse_px),
      {24, 38}, board.valid ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 80, 255));
    draw_label(
      image, sample_ready ? "READY: s save" : "click laser spot and keep full board visible",
      {24, 72}, sample_ready ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255));
    draw_label(
      image,
      allow_fit ? "s save | r clear | d delete | t fit | q quit"
                : "s save | r clear | d delete | q quit",
      {24, 106}, {230, 230, 230});
    if (point) {
      draw_label(
        image, fmt::format("point C: {:.3f}, {:.3f}, {:.3f} m", point->x(), point->y(), point->z()),
        {24, 140}, {255, 220, 80});
    }
    if (mouse.clicked_pixel && clicked_inside) {
      std::vector<cv::Point2f> undistorted;
      cv::undistortPoints(
        std::vector<cv::Point2f>{*mouse.clicked_pixel}, undistorted, config.camera.camera_matrix,
        config.camera.distort_coeffs);
      if (!undistorted.empty()) {
        draw_label(
          image,
          fmt::format("undistorted ray: [{:.6f}, {:.6f}, 1]", undistorted[0].x, undistorted[0].y),
          {24, 174}, {255, 220, 80});
      }
    }

    double scale = 1.0;
    cv::Mat display = fit_for_display(image, scale);
    mouse.scale = scale;
    mouse.display_width = display.cols;
    mouse.display_height = display.rows;
    cv::imshow(kWindowName, display);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q') break;
    if (key == 'r') mouse.clicked_pixel.reset();
    if (key == 'd' && !samples.empty()) {
      fs::remove(dataset / samples.back().image_path);
      tools::logger()->info("[LaserCalibration] Deleted sample {}", samples.back().id);
      samples.pop_back();
      save_raw_samples(dataset, samples);
    }
    if (key == 's') {
      if (!sample_ready) {
        tools::logger()->warn("[LaserCalibration] Sample is not ready");
        continue;
      }
      if (!samples.empty() && samples.front().image_size != raw_image.size()) {
        tools::logger()->error("[LaserCalibration] Image size changed; sample rejected");
        continue;
      }
      if (config.calibrated_image_size && raw_image.size() != *config.calibrated_image_size) {
        tools::logger()->error(
          "[LaserCalibration] Runtime image size does not match image_width/image_height");
        continue;
      }
      RawSample sample;
      sample.id = next_sample_id(samples);
      sample.image_path = fmt::format("images/{:04d}.jpg", sample.id);
      sample.laser_pixel = *mouse.clicked_pixel;
      sample.image_size = raw_image.size();
      sample.timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
      if (!cv::imwrite((dataset / sample.image_path).string(), raw_image)) {
        throw std::runtime_error("Failed to save calibration image");
      }
      samples.push_back(sample);
      save_raw_samples(dataset, samples);
      tools::logger()->info(
        "[LaserCalibration] Saved sample {} at ({:.2f}, {:.2f})", sample.id, sample.laser_pixel.x,
        sample.laser_pixel.y);
    }
    if (key == 't') {
      if (!allow_fit) {
        tools::logger()->warn("[LaserCalibration] Use --mode=capture-fit or --mode=fit");
      } else {
        fit_dataset(dataset, config, options, config_path, false);
      }
    }
  }
  return 0;
}

}  // namespace

const std::string keys =
  "{help h usage ?              |                              | 输出命令行帮助}"
  "{mode                         | fit                  | 运行模式：capture、fit 或 "
  "capture-fit}"
  "{config-path c                | ../configs/auto_drone.yaml | 标定配置 YAML 路径}"
  "{max-board-rmse-px           | 1.5                          | 棋盘格单帧最大重投影 RMS，单位 px}"
  "{ransac-threshold-mm          | 5.0                          | RANSAC 点到直线内点阈值，单位 mm}"
  "{ransac-iterations            | 2000                         | RANSAC 迭代次数}"
  "{bootstrap-iterations         | 200                          | Bootstrap 重采样次数}"
  "{min-samples                  | 15                           | 质量门禁要求的最少有效样本数}"
  "{min-bin-samples              | 3                            | 每个距离区间要求的最少样本数}"
  "{min-span-m                   | 0.5                          | 沿激光线方向的最小采样跨度，单位 "
  "m}"
  "{min-inlier-ratio             | 0.8                          | RANSAC 最小内点率}"
  "{max-line-rms-mm              | 5.0                          | 三维点到直线最大 RMS，单位 mm}"
  "{max-validation-rms-px       | 5.0                          | 验证集最大像素重投影 RMS，单位 px}"
  "{max-validation-p95-px       | 8.0                          | 验证集最大像素重投影 P95，单位 px}"
  "{max-bootstrap-direction-deg | 0.1                          | 激光方向 Bootstrap 最大 P95，单位 "
  "deg}"
  "{@dataset-folder              | ../assets/img_with_q         | 图片数据集及输出目录}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  try {
    const std::string mode = cli.get<std::string>("mode");
    const std::string config_path = cli.get<std::string>("config-path");
    const fs::path dataset = cli.get<std::string>(0);
    const CalibrationConfig config = load_config(config_path);
    const FitOptions options = options_from_cli(cli);
    if (mode == "fit") return fit_dataset(dataset, config, options, config_path, true);
    if (mode == "capture") {
      return capture_dataset(dataset, config_path, config, options, false);
    }
    if (mode == "capture-fit") {
      return capture_dataset(dataset, config_path, config, options, true);
    }
    throw std::runtime_error("Invalid mode: " + mode);
  } catch (const std::exception & error) {
    tools::logger()->error("[LaserCalibration] {}", error.what());
    return 1;
  }
}
