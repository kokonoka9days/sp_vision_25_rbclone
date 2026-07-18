#include "tasks/auto_drone/laser_ray_aim.hpp"

#include <fmt/core.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"

namespace
{

bool expect(bool condition, const std::string & message)
{
  if (!condition) fmt::print(stderr, "[FAIL] {}\n", message);
  return condition;
}

auto canonical_ray(Eigen::Vector3d origin, Eigen::Vector3d direction)
{
  direction.normalize();
  origin -= direction * direction.dot(origin);
  return auto_drone::LaserRay{origin, direction};
}

}  // namespace

int main()
{
  bool passed = true;

  const auto legacy_ray = canonical_ray(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
  const std::vector<Eigen::Vector3d> legacy_targets{
    {4.0, 1.0, 0.5}, {3.0, -2.0, -0.4}, {-2.0, 1.0, 0.3}};
  for (const auto & target : legacy_targets) {
    const auto solution = auto_drone::solve_laser_ray_aim(target, legacy_ray);
    passed &= expect(solution.has_value(), "legacy +X ray must be solvable");
    if (!solution) continue;
    const double expected_yaw = std::atan2(target.y(), target.x());
    const double expected_pitch = std::atan2(target.z(), target.head<2>().norm());
    passed &= expect(
      std::abs(tools::limit_rad(solution->yaw - expected_yaw)) < 1e-10,
      "zero-offset ray yaw must match the legacy atan2 result");
    passed &= expect(
      std::abs(solution->pitch - expected_pitch) < 1e-10,
      "zero-offset ray pitch must match the legacy atan2 result");
  }

  const auto calibrated_ray =
    canonical_ray(Eigen::Vector3d(0.015, 0.045, -0.025), Eigen::Vector3d(1.0, 0.018, 0.032));
  constexpr double expected_yaw = 0.43;
  constexpr double expected_pitch = 0.17;
  constexpr double expected_lambda = 3.4;
  const Eigen::Vector3d target =
    auto_drone::laser_command_rotation(expected_yaw, expected_pitch) *
    (calibrated_ray.origin_in_gimbal_m + expected_lambda * calibrated_ray.direction_in_gimbal);
  const auto calibrated_solution = auto_drone::solve_laser_ray_aim(target, calibrated_ray);
  passed &= expect(calibrated_solution.has_value(), "calibrated ray target must be solvable");
  if (calibrated_solution) {
    passed &= expect(
      std::abs(tools::limit_rad(calibrated_solution->yaw - expected_yaw)) < 1e-10,
      "calibrated ray yaw must recover the generating command");
    passed &= expect(
      std::abs(calibrated_solution->pitch - expected_pitch) < 1e-10,
      "calibrated ray pitch must recover the generating command");
    passed &= expect(
      std::abs(calibrated_solution->distance_along_ray_m - expected_lambda) < 1e-10,
      "calibrated ray must recover the positive distance along the beam");
    passed &= expect(
      calibrated_solution->residual_m < 1e-10,
      "closed-loop ray intersection residual must be negligible");

    const double legacy_yaw = std::atan2(target.y(), target.x());
    const double legacy_pitch = std::atan2(target.z(), target.head<2>().norm());
    const double correction = std::hypot(
      tools::limit_rad(calibrated_solution->yaw - legacy_yaw),
      calibrated_solution->pitch - legacy_pitch);
    passed &= expect(
      correction > 1e-3, "calibrated line must produce a measurable parallax/boresight correction");
  }

  const Eigen::Vector3d too_close =
    Eigen::Vector3d::UnitX() * (0.5 * calibrated_ray.origin_in_gimbal_m.norm());
  passed &= expect(
    !auto_drone::solve_laser_ray_aim(too_close, calibrated_ray).has_value(),
    "target sphere inside the laser-line offset must be rejected");

  auto invalid_ray = calibrated_ray;
  invalid_ray.direction_in_gimbal.setZero();
  passed &= expect(
    !auto_drone::solve_laser_ray_aim(target, invalid_ray).has_value(),
    "zero laser direction must be rejected");

  Eigen::Vector3d invalid_target = target;
  invalid_target.x() = std::numeric_limits<double>::quiet_NaN();
  passed &= expect(
    !auto_drone::solve_laser_ray_aim(invalid_target, calibrated_ray).has_value(),
    "non-finite target must be rejected");

  if (calibrated_solution) {
    fmt::print(
      "yaw={:.6f} pitch={:.6f} lambda={:.6f} residual={:.3e}m\n", calibrated_solution->yaw,
      calibrated_solution->pitch, calibrated_solution->distance_along_ray_m,
      calibrated_solution->residual_m);
  }
  return passed ? 0 : 1;
}
