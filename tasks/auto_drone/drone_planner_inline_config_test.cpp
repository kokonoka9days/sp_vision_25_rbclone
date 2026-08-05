#include "tasks/auto_drone/drone_planner.hpp"

#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
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

int main(int argc, char ** argv)
{
  if (argc == 2) {
    try {
      auto_drone::Planner planner(argv[1]);
      const auto laser_reference = planner.laser_reference_pixel(12.806);
      if (!laser_reference) {
        fmt::print(stderr, "[FAIL] configured laser reference is unavailable\n");
        return 1;
      }
      fmt::print(
        "configured laser reference at 12.806 m: ({:.3f}, {:.3f}) px\n",
        laser_reference->x(), laser_reference->y());
      return 0;
    } catch (const std::exception & error) {
      fmt::print(stderr, "[FAIL] configured laser ray was rejected: {}\n", error.what());
      return 1;
    }
  }

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
camera_matrix: [14955.275481254272, 0.0, 424.5275247834595, 0.0, 14917.483385201438, 890.1436041583838, 0.0, 0.0, 1.0]
distort_coeffs: [0.259455463319462, 7.781591144598162, 0.014803400834467228, -0.017879580487416027, 0.15829957121310623]
xyz_offset: [0.0, 0.0, 0.0]
fire_thresh: 0.1
gimbal_control_delay: 0.02
max_yaw_acc: 50.0
Q_yaw: [9000000.0, 0.0]
R_yaw: [1.0]
max_pitch_acc: 100.0
Q_pitch: [9000000.0, 0.0]
R_pitch: [1.0]
R_camera2gimbal: [-0.040421293314026, 0.324421929857806, 0.945048427595651, -0.975158375276931, -0.218967173836949, 0.033459227563082, 0.217789490504245, -0.920219423947863, 0.325213698381750]
t_camera2gimbal: [0.045, 0.0, -0.050]
laser_ray_enabled: true
laser_line_origin_in_camera_m: [-0.0019666593140683745, 0.0001422488863406174, 0.000060260134027797516]
laser_line_direction_in_camera: [0.029041167184577103, -0.021836797894733594, 0.9993396644115864]
)";
  output.close();

  try {
    auto_drone::Planner planner(config.path().string());
    const auto laser_reference = planner.laser_reference_pixel(12.806);
    if (
      !laser_reference || std::abs(laser_reference->x() - 855.91) > 0.1 ||
      std::abs(laser_reference->y() - 565.07) > 0.1) {
      fmt::print(
        stderr, "[FAIL] calibrated laser reference pixel is wrong: ({:.3f}, {:.3f})\n",
        laser_reference ? laser_reference->x() : std::numeric_limits<double>::quiet_NaN(),
        laser_reference ? laser_reference->y() : std::numeric_limits<double>::quiet_NaN());
      return 1;
    }
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
