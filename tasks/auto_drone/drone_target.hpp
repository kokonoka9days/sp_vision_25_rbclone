#ifndef AUTO_DRONE__TARGET_HPP
#define AUTO_DRONE__TARGET_HPP

#include <Eigen/Dense>
#include "drone_armor.hpp"          
#include "tools/extended_kalman_filter.hpp"      

namespace auto_drone
{

class Target
{
public:
  Target() = default;
  Target(const Drone & drone);

  void predict(double dt);
  void update(const Drone & drone);

  Eigen::Vector3d get_xyz() const;
  Eigen::Vector3d get_v() const;

  DroneName name;
  Color color;

  // 修改：tools 的 EKF 不是模板类，去除 <6, 3>
  tools::ExtendedKalmanFilter ekf_;
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__TARGET_HPP