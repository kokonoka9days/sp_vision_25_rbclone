#include "tasks/auto_drone/drone_planner.hpp"

#include <fmt/core.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{

class TemporaryConfig
{
public:
  TemporaryConfig()
  {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("drone_planner_inline_laser_" + std::to_string(suffix) + ".yaml");
  }

  ~TemporaryConfig()
  {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path & path() const { return path_; }

private:
  std::filesystem::path path_;
};

}  // namespace

int main()
{
  TemporaryConfig config;
  std::ofstream output(config.path());
  output << R"(
yaw_offset: 0.0
pitch_offset: 0.0
visual_servo_enabled: true
visual_servo_kp: 0.65
visual_servo_ki: 0.8
visual_servo_max_correction_deg: 0.6
visual_servo_error_limit_px: 200.0
visual_servo_deadband_px: 1.5
xyz_offset: [0.0, 0.0, 0.0]
fire_thresh: 0.1
gimbal_control_delay: 0.02
max_yaw_acc: 50.0
Q_yaw: [9000000.0, 0.0]
R_yaw: [1.0]
max_pitch_acc: 100.0
Q_pitch: [9000000.0, 0.0]
R_pitch: [1.0]
R_camera2gimbal: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
t_camera2gimbal: [0.0, 0.0, 0.0]
laser_ray_enabled: true
laser_line_origin_in_camera_m: [0.0, 0.01, -0.02]
laser_line_direction_in_camera: [1.0, 0.0, 0.0]
)";
  output.close();

  try {
    auto_drone::Planner planner(config.path().string());
    const auto timestamp = std::chrono::steady_clock::now();
    planner.update_visual_feedback(100.0, -50.0, timestamp);
    if (
      planner.visual_yaw_correction_deg() >= 0.0 ||
      planner.visual_pitch_correction_deg() <= 0.0) {
      fmt::print(stderr, "[FAIL] visual feedback correction signs are wrong\n");
      return 1;
    }
  } catch (const std::exception & error) {
    fmt::print(stderr, "[FAIL] inline laser configuration was rejected: {}\n", error.what());
    return 1;
  }

  fmt::print("inline laser configuration loaded without laser_ray_config_path\n");
  return 0;
}
