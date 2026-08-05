#include "laser_ray_aim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"

namespace auto_drone
{
namespace
{

constexpr double kGeometryEpsilon = 1e-9;
constexpr double kPi = 3.14159265358979323846;

bool finite_vector(const Eigen::Vector3d & value) { return value.array().isFinite().all(); }

double angular_distance(double lhs, double rhs) { return std::abs(tools::limit_rad(lhs - rhs)); }

}  // namespace

Eigen::Matrix3d laser_command_rotation(double yaw, double pitch)
{
  const double cos_yaw = std::cos(yaw);
  const double sin_yaw = std::sin(yaw);
  const double cos_pitch = std::cos(pitch);
  const double sin_pitch = std::sin(pitch);

  Eigen::Matrix3d rotation;
  rotation << cos_yaw * cos_pitch, -sin_yaw, -cos_yaw * sin_pitch, sin_yaw * cos_pitch, cos_yaw,
    -sin_yaw * sin_pitch, sin_pitch, 0.0, cos_pitch;
  return rotation;
}

std::optional<LaserAimSolution> solve_laser_ray_aim(
  const Eigen::Vector3d & target_in_world_m, const LaserRay & laser_ray)
{
  if (
    !finite_vector(target_in_world_m) || !finite_vector(laser_ray.origin_in_gimbal_m) ||
    !finite_vector(laser_ray.direction_in_gimbal)) {
    return std::nullopt;
  }

  const double direction_norm = laser_ray.direction_in_gimbal.norm();
  const double target_distance = target_in_world_m.norm();
  if (direction_norm < kGeometryEpsilon || target_distance < kGeometryEpsilon) {
    return std::nullopt;
  }

  const Eigen::Vector3d direction = laser_ray.direction_in_gimbal / direction_norm;
  const Eigen::Vector3d origin =
    laser_ray.origin_in_gimbal_m - direction * direction.dot(laser_ray.origin_in_gimbal_m);

  double lambda_squared = target_distance * target_distance - origin.squaredNorm();
  const double squared_tolerance =
    kGeometryEpsilon * std::max(1.0, target_distance * target_distance);
  if (lambda_squared < -squared_tolerance) return std::nullopt;
  lambda_squared = std::max(0.0, lambda_squared);
  const double lambda = std::sqrt(lambda_squared);
  const Eigen::Vector3d point_on_ray = origin + lambda * direction;

  const double pitch_plane_radius = std::hypot(point_on_ray.x(), point_on_ray.z());
  if (pitch_plane_radius < kGeometryEpsilon) return std::nullopt;

  const double pitch_ratio = target_in_world_m.z() / pitch_plane_radius;
  if (std::abs(pitch_ratio) > 1.0 + kGeometryEpsilon) return std::nullopt;

  const double clamped_ratio = std::clamp(pitch_ratio, -1.0, 1.0);
  const double pitch_phase = std::atan2(point_on_ray.z(), point_on_ray.x());
  const double principal = std::asin(clamped_ratio);
  const std::array<double, 2> pitch_candidates{
    tools::limit_rad(principal - pitch_phase), tools::limit_rad(kPi - principal - pitch_phase)};
  const double nominal_pitch =
    std::atan2(target_in_world_m.z(), std::hypot(target_in_world_m.x(), target_in_world_m.y()));

  std::optional<LaserAimSolution> best;
  double best_score = std::numeric_limits<double>::infinity();
  for (const double pitch : pitch_candidates) {
    const double x_after_pitch =
      std::cos(pitch) * point_on_ray.x() - std::sin(pitch) * point_on_ray.z();
    const double source_azimuth = std::atan2(point_on_ray.y(), x_after_pitch);
    const double target_azimuth = std::atan2(target_in_world_m.y(), target_in_world_m.x());
    const double yaw = tools::limit_rad(target_azimuth - source_azimuth);
    const double residual =
      (laser_command_rotation(yaw, pitch) * point_on_ray - target_in_world_m).norm();
    const double residual_tolerance = 1e-7 * std::max(1.0, target_distance);
    if (!std::isfinite(residual) || residual > residual_tolerance) continue;

    const double score = angular_distance(pitch, nominal_pitch);
    if (score < best_score) {
      best_score = score;
      best = LaserAimSolution{yaw, pitch, lambda, residual};
    }
  }
  return best;
}

}  // namespace auto_drone
