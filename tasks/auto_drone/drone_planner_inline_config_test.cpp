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
visual_servo_max_correction_rate_deg_s: 1.5
visual_servo_error_limit_px: 200.0
visual_servo_deadband_px: 1.5
visual_servo_integral_speed_threshold_mps: 0.08
visual_servo_integral_error_limit_px: 40.0
visual_servo_integral_settle_time_s: 0.5
visual_servo_yaw_correction_per_yaw: -0.086
visual_servo_pitch_correction_per_yaw: 0.0686
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
    planner.update_visual_feedback(100.0, -50.0, 1.0, 0.0, timestamp);
    if (
      planner.visual_yaw_correction_deg() >= 0.0 ||
      planner.visual_pitch_correction_deg() <= 0.0) {
      fmt::print(stderr, "[FAIL] visual feedback correction signs are wrong\n");
      return 1;
    }
    if (planner.visual_integrator_active()) {
      fmt::print(stderr, "[FAIL] visual integrator was active for a moving target\n");
      return 1;
    }
    planner.update_visual_feedback(
      5.0, -5.0, 0.0, 0.0, timestamp + std::chrono::milliseconds(33));
    if (planner.visual_integrator_active()) {
      fmt::print(stderr, "[FAIL] visual integrator activated before the settle dwell elapsed\n");
      return 1;
    }
    planner.update_visual_feedback(
      5.0, -5.0, 0.0, 0.0, timestamp + std::chrono::milliseconds(600));
    if (!planner.visual_integrator_active()) {
      fmt::print(stderr, "[FAIL] visual integrator did not activate after the settle dwell\n");
      return 1;
    }
    planner.update_visual_feedback(
      0.0, 0.0, 1.0, 1.0 / 57.3, timestamp + std::chrono::milliseconds(633));
    if (
      planner.visual_yaw_angle_feedforward_deg() >= 0.0 ||
      planner.visual_pitch_angle_feedforward_deg() <= 0.0) {
      fmt::print(stderr, "[FAIL] relative-yaw feedforward signs are wrong\n");
      return 1;
    }
  } catch (const std::exception & error) {
    fmt::print(stderr, "[FAIL] inline laser configuration was rejected: {}\n", error.what());
    return 1;
  }

  fmt::print("inline laser configuration loaded without laser_ray_config_path\n");
  return 0;
}
