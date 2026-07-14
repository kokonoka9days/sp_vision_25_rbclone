#include <cmath>
#include <fmt/core.h>
#include <string>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_target.hpp"

namespace
{
auto_buff::PowerRune make_observation(
  double phase, int track_id, const Eigen::Vector3d & center,
  const Eigen::Vector3d & plane_normal)
{
  Eigen::Vector3d zero_axis =
    Eigen::Vector3d::UnitZ() - plane_normal.z() * plane_normal;
  zero_axis.normalize();
  const Eigen::Vector3d quarter_axis = plane_normal.cross(zero_axis).normalized();
  const Eigen::Vector3d radial =
    std::cos(phase) * zero_axis + std::sin(phase) * quarter_axis;

  auto_buff::PowerRune p;
  p.track_id = track_id;
  p.pose_quality = auto_buff::BuffPoseQuality::FULL_8_POINT;
  p.measurement_noise_scale = 1.0;
  p.xyz_in_world = center;
  p.ypd_in_world = tools::xyz2ypd(center);
  p.blade_xyz_in_world = center + auto_buff::RUNE_RADIUS_M * radial;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);
  p.plane_normal_in_world = plane_normal;
  return p;
}

bool near(double actual, double expected, double tolerance, const char * name)
{
  if (std::abs(actual - expected) <= tolerance) return true;
  fmt::print(stderr, "{}: actual {:.6f}, expected {:.6f}, tolerance {:.6f}\n",
    name, actual, expected, tolerance);
  return false;
}
}  // namespace

int main(int argc, char * argv[])
{
  const Eigen::Vector3d center(5.8, 0.3, 0.5);
  const Eigen::Vector3d plane_normal = Eigen::Vector3d(-0.91, 0.18, 0.37).normalized();
  const auto start = std::chrono::steady_clock::now();
  constexpr double dt = 0.01;
  constexpr double omega = CV_PI / 3.0;
  bool ok = true;

  auto_buff::SmallTarget startup_target;
  double startup_phase = 0.4;
  for (int i = 0; i < 4; ++i) {
    startup_phase += omega * dt;
    auto observation = make_observation(startup_phase, 100, center, plane_normal);
    auto timestamp = start + std::chrono::milliseconds(i * 10);
    startup_target.get_target(observation, timestamp);
    if (i == 0) {
      ok &= startup_target.can_control();
      ok &= !startup_target.prediction_ready();
    }
  }
  ok &= startup_target.prediction_ready();

  auto reverse_outlier = make_observation(
    startup_phase - omega * dt, 100, center, plane_normal);
  auto reverse_timestamp = start + std::chrono::milliseconds(40);
  startup_target.get_target(reverse_outlier, reverse_timestamp);
  auto latched_direction = startup_target.clone();
  latched_direction->predict(0.05);
  ok &= latched_direction->ekf_x()[5] > startup_target.ekf_x()[5];

  auto_buff::SmallTarget recovery_target;
  double recovery_phase = 1.0;
  for (int i = 0; i < 4; ++i) {
    recovery_phase += omega * dt;
    auto observation = make_observation(recovery_phase, 200, center, plane_normal);
    auto timestamp = start + std::chrono::milliseconds(i * 10);
    recovery_target.get_target(observation, timestamp);
  }
  const double stable_phase = recovery_target.ekf_x()[5];
  auto jumped = make_observation(recovery_phase + 1.0, 200, center, plane_normal);
  auto jumped_timestamp = start + std::chrono::milliseconds(40);
  recovery_target.get_target(jumped, jumped_timestamp);
  ok &= std::abs(recovery_target.ekf_x()[5] - stable_phase) < 0.2;
  auto coherent_jump = make_observation(recovery_phase + 1.02, 200, center, plane_normal);
  auto recovery_timestamp = start + std::chrono::milliseconds(50);
  recovery_target.get_target(coherent_jump, recovery_timestamp);
  ok &= std::abs(recovery_target.ekf_x()[5] - (recovery_phase + 1.02)) < 0.1;

  auto short_dropout = start + std::chrono::milliseconds(80);
  startup_target.get_target(std::nullopt, short_dropout);
  ok &= startup_target.can_control();
  ok &= startup_target.is_blind();
  ok &= !startup_target.can_fire(short_dropout);

  auto control_timeout = start + std::chrono::milliseconds(160);
  startup_target.get_target(std::nullopt, control_timeout);
  ok &= !startup_target.can_control();
  auto retention_timeout = start + std::chrono::milliseconds(500);
  startup_target.get_target(std::nullopt, retention_timeout);
  ok &= startup_target.reset_count() == 1;

  auto_buff::SmallTarget target;
  double phase = 2.8;

  for (int i = 0; i < 220; ++i) {
    phase += omega * dt;
    const int track_id = i < 110 ? 1 : 2;
    if (i == 110) phase += auto_buff::RUNE_SLOT_ANGLE;
    auto observation = make_observation(phase, track_id, center, plane_normal);
    auto timestamp = start + std::chrono::microseconds(static_cast<int64_t>(i * dt * 1e6));
    target.get_target(observation, timestamp);
  }

  ok &= !target.is_unsolve();
  ok &= target.reset_count() == 0;
  ok &= near(target.ekf_x()[5], phase, 0.03, "continuous phase");

  const Eigen::Vector3d filtered_center =
    tools::ypd2xyz(Eigen::Vector3d(target.ekf_x()[0], target.ekf_x()[2], target.ekf_x()[3]));
  ok &= near((filtered_center - center).norm(), 0.0, 0.01, "filtered center");

  auto predicted = target.clone();
  predicted->predict(0.1);
  ok &= near(predicted->ekf_x()[5] - target.ekf_x()[5], omega * 0.1, 1e-4,
    "prediction phase delta");

  const std::string config = argc > 1 ? argv[1] : "../configs/xiaohei.yaml";
  auto_buff::Aimer aimer(config);
  io::GimbalState gimbal_state{};
  gimbal_state.bullet_speed = 24.0f;
  auto aim_timestamp = start + std::chrono::milliseconds(2200);
  const auto plan = aimer.mpc_aim(target, aim_timestamp, gimbal_state, false);
  const auto * aim_prediction = aimer.predicted_target();
  ok &= plan.control;
  ok &= aim_prediction != nullptr;
  if (aim_prediction != nullptr) {
    ok &= aim_prediction->ekf_x()[5] > target.ekf_x()[5];
  }

  const Eigen::Vector3d predicted_blade =
    predicted->point_buff2world(Eigen::Vector3d(0.0, 0.0, auto_buff::RUNE_RADIUS_M));
  ok &= near((predicted_blade - filtered_center).norm(), auto_buff::RUNE_RADIUS_M, 1e-6,
    "prediction radius");

  if (!ok) return 1;
  fmt::print("auto_buff_small_target_test passed\n");
  return 0;
}
