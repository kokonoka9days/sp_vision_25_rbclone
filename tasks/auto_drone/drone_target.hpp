#ifndef AUTO_DRONE__TARGET_HPP
#define AUTO_DRONE__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include "drone_armor.hpp"          
#include "tools/extended_kalman_filter.hpp"      

namespace auto_drone
{

class Target
{
public:
  Target() = default;
  Target(const Drone & drone, std::chrono::steady_clock::time_point timestamp);

  void predict(double dt);
  void update(const Drone & drone, std::chrono::steady_clock::time_point timestamp);

  void set_state_timestamp(std::chrono::steady_clock::time_point timestamp);
  std::chrono::steady_clock::time_point state_timestamp() const;
  std::chrono::steady_clock::time_point last_observation_timestamp() const;

  Eigen::Vector3d get_xyz() const;
  Eigen::Vector3d get_v() const;

  DroneName name;
  Color color;

  // 修改：tools 的 EKF 不是模板类，去除 <6, 3>
  tools::ExtendedKalmanFilter ekf_;

private:
  std::chrono::steady_clock::time_point state_timestamp_;
  std::chrono::steady_clock::time_point last_observation_timestamp_;
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__TARGET_HPP
