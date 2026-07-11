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
  auto_buff::SmallTarget target;
  const Eigen::Vector3d center(5.8, 0.3, 0.5);
  const Eigen::Vector3d plane_normal = Eigen::Vector3d(-0.91, 0.18, 0.37).normalized();
  const auto start = std::chrono::steady_clock::now();
  constexpr double dt = 0.01;
  constexpr double omega = CV_PI / 3.0;
  double phase = 2.8;

  for (int i = 0; i < 220; ++i) {
    phase += omega * dt;
    const int track_id = i < 110 ? 1 : 2;
    if (i == 110) phase += auto_buff::RUNE_SLOT_ANGLE;
    auto observation = make_observation(phase, track_id, center, plane_normal);
    auto timestamp = start + std::chrono::microseconds(static_cast<int64_t>(i * dt * 1e6));
    target.get_target(observation, timestamp);
  }

  bool ok = true;
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

  const Eigen::Vector3d predicted_blade =
    predicted->point_buff2world(Eigen::Vector3d(0.0, 0.0, auto_buff::RUNE_RADIUS_M));
  ok &= near((predicted_blade - filtered_center).norm(), auto_buff::RUNE_RADIUS_M, 1e-6,
    "prediction radius");

  if (!ok) return 1;
  fmt::print("auto_buff_small_target_test passed\n");
  return 0;
}
