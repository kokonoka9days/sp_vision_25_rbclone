#include <cmath>
#include <fmt/core.h>

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

int main()
{
  auto_buff::BigTarget target;
  const Eigen::Vector3d center(6.2, -0.2, 0.7);
  const Eigen::Vector3d plane_normal = Eigen::Vector3d(-0.88, -0.21, 0.43).normalized();
  const auto start = std::chrono::steady_clock::now();
  constexpr double dt = 0.01;
  constexpr double speed = 1.3;
  double phase = 2.9;

  // Keep the sample count below the sine-fit readiness threshold to test fallback prediction.
  for (int i = 0; i < 50; ++i) {
    phase += speed * dt;
    auto observation = make_observation(phase, 1, center, plane_normal);
    auto timestamp = start + std::chrono::microseconds(static_cast<int64_t>(i * dt * 1e6));
    target.get_target(observation, timestamp);
  }

  bool ok = true;
  ok &= !target.is_unsolve();
  ok &= target.reset_count() == 0;
  ok &= near(target.ekf_x()[5], phase, 0.04, "continuous phase");
  ok &= near(std::abs(target.ekf_x()[6]), speed, 0.15, "fallback speed");

  auto predicted = target.clone();
  predicted->predict(0.1);
  const double prediction_delta = predicted->ekf_x()[5] - target.ekf_x()[5];
  ok &= prediction_delta > 0.08 && prediction_delta < 0.18;

  phase += auto_buff::RUNE_SLOT_ANGLE;
  auto switched_observation = make_observation(phase, 2, center, plane_normal);
  auto switched_timestamp = start + std::chrono::milliseconds(510);
  target.get_target(switched_observation, switched_timestamp);
  ok &= target.reset_count() == 0;

  const Eigen::Vector3d filtered_center =
    tools::ypd2xyz(Eigen::Vector3d(target.ekf_x()[0], target.ekf_x()[2], target.ekf_x()[3]));
  ok &= near((filtered_center - center).norm(), 0.0, 0.02, "center after track switch");

  if (!ok) return 1;
  fmt::print("auto_buff_big_target_test passed\n");
  return 0;
}
