#ifndef AUTO_BUFF__RP_CORE__CONFIG_HPP
#define AUTO_BUFF__RP_CORE__CONFIG_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

class RuneConfigNode
{
public:
  RuneConfigNode() = default;
  explicit RuneConfigNode(YAML::Node node) : node_(std::move(node)) {}

  RuneConfigNode operator[](const char * key) const { return RuneConfigNode(node_[key]); }
  RuneConfigNode operator[](const std::string & key) const { return RuneConfigNode(node_[key]); }

  template<typename T, typename = std::enable_if_t<!std::is_same_v<T, RuneConfigNode>>>
  operator T() const
  {
    if (!node_ || !node_.IsDefined()) {
      throw std::runtime_error("missing power_rune configuration value");
    }
    return node_.as<T>();
  }

private:
  YAML::Node node_;
};

class PowerRuneConfigStore
{
public:
  void initialize(
    const std::filesystem::path & defaults_path,
    const std::filesystem::path & robot_config_path);
  void updateJson() {}

  void set_bullet_speed(double bullet_speed);
  void set_image_size(int width, int height);

  const cv::Matx33d & camera_matrix() const { return camera_matrix_; }
  const cv::Matx<double, 1, 5> & distortion() const { return distortion_; }
  const Eigen::Matrix3d & R_gimbal2imubody() const { return R_gimbal2imubody_; }
  const Eigen::Matrix3d & R_camera2gimbal() const { return R_camera2gimbal_; }
  const Eigen::Vector3d & t_camera2gimbal() const { return t_camera2gimbal_; }
  const std::filesystem::path & onnx_path() const { return onnx_path_; }
  const std::filesystem::path & engine_path() const { return engine_path_; }
  double yaw_offset_rad() const { return yaw_offset_rad_; }
  double pitch_offset_rad() const { return pitch_offset_rad_; }

  RuneConfigNode config_;

private:
  YAML::Node root_;
  cv::Matx33d camera_matrix_ = cv::Matx33d::eye();
  cv::Matx<double, 1, 5> distortion_{};
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  std::filesystem::path onnx_path_;
  std::filesystem::path engine_path_;
  double yaw_offset_rad_ = 0.0;
  double pitch_offset_rad_ = 0.0;
};

extern PowerRuneConfigStore J_POWER_RUNE;

#endif
