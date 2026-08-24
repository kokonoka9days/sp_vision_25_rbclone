#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

PowerRuneConfigStore J_POWER_RUNE;

namespace
{
constexpr double kPi = 3.14159265358979323846;
void merge_yaml(YAML::Node target, const YAML::Node & overrides)
{
  if (!overrides || !overrides.IsMap()) return;
  for (const auto & item : overrides) {
    const std::string key = item.first.as<std::string>();
    const YAML::Node value = item.second;
    if (value.IsMap() && target[key] && target[key].IsMap()) {
      merge_yaml(target[key], value);
    } else {
      target[key] = YAML::Clone(value);
    }
  }
}

std::filesystem::path resolve_path(
  const std::filesystem::path & base_file, const std::string & value)
{
  std::filesystem::path path(value);
  if (path.is_relative()) path = base_file.parent_path() / path;
  return path.lexically_normal();
}

std::vector<double> require_vector(const YAML::Node & root, const char * key, std::size_t size)
{
  if (!root[key]) throw std::runtime_error(std::string("missing robot config key: ") + key);
  const auto values = root[key].as<std::vector<double>>();
  if (values.size() != size) {
    throw std::runtime_error(
      std::string("robot config key '") + key + "' must contain " + std::to_string(size) +
      " values");
  }
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(std::string("robot config key '") + key + "' contains non-finite data");
    }
  }
  return values;
}

Eigen::Matrix3d matrix3(const std::vector<double> & values)
{
  return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(values.data());
}

double require_finite_scalar(
  const YAML::Node & root, const char * section, const char * key,
  double minimum, double maximum)
{
  const YAML::Node value_node = root[section][key];
  if (!value_node) {
    throw std::runtime_error(
      std::string("missing power_rune config key: ") + section + "." + key);
  }
  const double value = value_node.as<double>();
  if (!std::isfinite(value) || value < minimum || value > maximum) {
    throw std::runtime_error(
      std::string("power_rune config key out of range: ") + section + "." + key);
  }
  return value;
}
}  // namespace

void PowerRuneConfigStore::initialize(
  const std::filesystem::path & defaults_path,
  const std::filesystem::path & robot_config_path)
{
  root_ = YAML::LoadFile(defaults_path.string());
  YAML::Node robot = YAML::LoadFile(robot_config_path.string());
  if (robot["power_rune"]) merge_yaml(root_, robot["power_rune"]);

  const auto camera = require_vector(robot, "camera_matrix", 9);
  const auto distortion = require_vector(robot, "distort_coeffs", 5);
  const auto R_gimbal2imu = require_vector(robot, "R_gimbal2imubody", 9);
  const auto R_camera2gimbal = require_vector(robot, "R_camera2gimbal", 9);
  const auto t_camera2gimbal = require_vector(robot, "t_camera2gimbal", 3);

  camera_matrix_ = cv::Matx33d(camera.data());
  distortion_ = cv::Matx<double, 1, 5>(distortion.data());
  R_gimbal2imubody_ = matrix3(R_gimbal2imu);
  R_camera2gimbal_ = matrix3(R_camera2gimbal);
  t_camera2gimbal_ = Eigen::Map<const Eigen::Vector3d>(t_camera2gimbal.data());

  const double yaw_offset_degree = robot["yaw_offset"]
    ? robot["yaw_offset"].as<double>()
    : root_["offset"]["yaw"].as<double>();
  const double pitch_offset_degree = robot["pitch_offset"]
    ? robot["pitch_offset"].as<double>()
    : root_["offset"]["pitch"].as<double>();
  yaw_offset_rad_ = yaw_offset_degree * kPi / 180.0;
  pitch_offset_rad_ = pitch_offset_degree * kPi / 180.0;
  root_["offset"]["yaw"] = 0.0;
  root_["offset"]["pitch"] = 0.0;

  const YAML::Node detector = root_["detector"];
  onnx_path_ = resolve_path(defaults_path, detector["onnx_path"].as<std::string>());
  engine_path_ = resolve_path(defaults_path, detector["engine_path"].as<std::string>());
  if (robot["power_rune"] && robot["power_rune"]["detector"]) {
    const YAML::Node override_detector = robot["power_rune"]["detector"];
    if (override_detector["onnx_path"]) {
      onnx_path_ = resolve_path(robot_config_path, override_detector["onnx_path"].as<std::string>());
    }
    if (override_detector["engine_path"]) {
      engine_path_ = resolve_path(robot_config_path, override_detector["engine_path"].as<std::string>());
    }
  }

  if (!(camera_matrix_(0, 0) > 0.0 && camera_matrix_(1, 1) > 0.0)) {
    throw std::runtime_error("camera focal lengths must be positive");
  }
  if (std::abs(R_camera2gimbal_.determinant() - 1.0) > 0.05) {
    throw std::runtime_error("R_camera2gimbal must be a proper rotation matrix");
  }
  if (std::abs(R_gimbal2imubody_.determinant() - 1.0) > 0.05) {
    throw std::runtime_error("R_gimbal2imubody must be a proper rotation matrix");
  }

  require_finite_scalar(root_, "rune_ballistic_model", "radius", 0.1, 2.0);
  require_finite_scalar(root_, "rune_ballistic_model", "bullet_flying_speed", 10.0, 35.0);
  require_finite_scalar(root_, "rune_ballistic_model", "grivaty", 5.0, 15.0);
  require_finite_scalar(root_, "rune_ballistic_model", "flying_time_max", 0.05, 5.0);
  require_finite_scalar(root_, "rune_ballistic_model", "delay_time", 0.0, 1.0);
  require_finite_scalar(root_, "small_phase_estimate", "max_data_interval", 0.001, 1.0);
  require_finite_scalar(root_, "small_phase_estimate", "min_size_to_confirm_rotation", 2.0, 1000.0);
  require_finite_scalar(root_, "big_phase_motion_estimate", "min_data_size_to_build_motion_model", 5.0, 5000.0);
  const double omega_lower = require_finite_scalar(
    root_, "big_phase_motion_estimate", "omega_lower_bound", 0.01, 10.0);
  const double omega_upper = require_finite_scalar(
    root_, "big_phase_motion_estimate", "omega_upper_bound", 0.01, 10.0);
  if (omega_lower >= omega_upper) {
    throw std::runtime_error("big rune omega_lower_bound must be below omega_upper_bound");
  }

  config_ = RuneConfigNode(root_);
}

void PowerRuneConfigStore::set_bullet_speed(double bullet_speed)
{
  root_["rune_ballistic_model"]["bullet_flying_speed"] = bullet_speed;
  config_ = RuneConfigNode(root_);
}

void PowerRuneConfigStore::set_image_size(int width, int height)
{
  root_["camera"]["width"] = width;
  root_["camera"]["height"] = height;
  root_["project"]["max_img_width"] =
    std::max(width, root_["project"]["max_img_width"].as<int>());
  root_["project"]["max_img_hight"] =
    std::max(height, root_["project"]["max_img_hight"].as<int>());
  config_ = RuneConfigNode(root_);
}
