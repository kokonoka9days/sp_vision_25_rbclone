// rv_from_fyt.hpp
#ifndef STATE2EST_HPP
#define STATE2EST_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <utility>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
/// @brief 状态估计接口
class State2Est : public tools::ExtendedKalmanFilter
{
public:
  using tools::ExtendedKalmanFilter::ExtendedKalmanFilter;

  virtual ~State2Est() = default;
    
  /// @brief         卡尔曼更新
  /// @param dt      采集间隔
  /// @param u       控制量
  /// @param noises  噪声
  virtual void kf_predict(double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises) = 0;

  /// @brief         mpc预测
  /// @param dt      采集间隔
  /// @param u       控制量
  /// @param noises  噪声
  virtual void mpc_predict(double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises) = 0;
};
}  // namespace auto_aim


#endif  // STATE2EST_HPP
