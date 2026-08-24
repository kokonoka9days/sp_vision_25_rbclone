#ifndef AUTO_BUFF__RP_CORE__RUNTIME_TYPES_HPP
#define AUTO_BUFF__RP_CORE__RUNTIME_TYPES_HPP

#include <chrono>
#include <cstdint>

#include <Eigen/Dense>

using RuneTimestamp = std::chrono::steady_clock::time_point;

struct CameraPose
{
  Eigen::Matrix3d R_car_from_camera = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_car_from_camera = Eigen::Vector3d::Zero();
};

namespace rune_time
{
inline uint64_t to_nanoseconds(const RuneTimestamp & timestamp)
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count());
}

inline RuneTimestamp from_nanoseconds(uint64_t nanoseconds)
{
  return RuneTimestamp(std::chrono::nanoseconds(nanoseconds));
}
}  // namespace rune_time

#endif
