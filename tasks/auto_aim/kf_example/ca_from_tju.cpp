// ca_from_tju.cpp
#include "ca_from_tju.hpp"

#include <cmath>
#include <stdexcept>

#include "tools/math_tools.hpp"

namespace auto_aim
{

/**
 * @brief 构造函数：验证维度并初始化基类
 */
CAFromTJU::CAFromTJU(const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0)
: State2Est(x0, P0, state_add)
{
  if (
    x0.size() != kStateDimension || P0.rows() != kStateDimension ||
    P0.cols() != kStateDimension) {
    throw std::invalid_argument("CAFromTJU requires an 11-dimensional state");
  }
}

/**
 * @brief 卡尔曼滤波预测接口（调用 predict_model）
 */
void CAFromTJU::kf_predict(
  double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises)
{
  // TrackQueueV3 为自主模型，加速度已包含在状态中，故忽略输入 u
  (void)u;
  predict_model(dt, noises);
}

/**
 * @brief MPC预测接口（直接复用 kf_predict）
 */
void CAFromTJU::mpc_predict(
  double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises)
{
  kf_predict(dt, u, noises);
}

/**
 * @brief 执行常加速度模型的预测步骤
 * 
 * 构建状态转移矩阵 F 和过程噪声协方差 Q，然后调用基类的 predict 方法。
 * 状态转移方程（连续时间常加速度离散化）：
 *   x_new = x + v*dt + 0.5*a*dt^2
 *   v_new = v + a*dt
 *   a_new = a
 * 偏航角及其速度和加速度同理。
 */
void CAFromTJU::predict_model(
  double dt, const Eigen::VectorXd & process_noise_diagonal)
{
  // 参数有效性检查
  if (!std::isfinite(dt) || dt < 0.0) {
    throw std::invalid_argument("CAFromTJU requires a finite, non-negative dt");
  }
  if (process_noise_diagonal.size() != kStateDimension) {
    throw std::invalid_argument("CAFromTJU requires 11 process-noise values");
  }
  if (
    !process_noise_diagonal.allFinite() ||
    (process_noise_diagonal.array() < 0.0).any()) {
    throw std::invalid_argument(
      "CAFromTJU process-noise values must be finite and non-negative");
  }

  const double half_dt_squared = 0.5 * dt * dt;
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateDimension, kStateDimension);

  // 位置状态索引：x=0, y=1, z=2, yaw=3
  // 速度状态索引：vx=4, vy=5, vz=6, vyaw=7
  // 加速度状态索引：ax=8, ay=9, ayaw=10
  // 设定 F 中各元素：x = x + vx*dt + ax*0.5*dt^2
  F(0, 4) = dt;
  F(0, 8) = half_dt_squared;
  F(1, 5) = dt;
  F(1, 9) = half_dt_squared;
  F(2, 6) = dt;
  F(3, 7) = dt;
  F(3, 10) = half_dt_squared;  // yaw = yaw + vyaw*dt + ayaw*0.5*dt^2
  // 速度更新：vx = vx + ax*dt
  F(4, 8) = dt;
  F(5, 9) = dt;
  F(7, 10) = dt;  // vyaw = vyaw + ayaw*dt
  // 加速度自身不变（恒加速度模型）

  // 过程噪声协方差 Q = diag(process_noise_diagonal)
  const Eigen::MatrixXd Q = process_noise_diagonal.asDiagonal();

  // 定义非线性预测函数（线性，但偏航角需归一化）
  auto transition = [&F](const Eigen::VectorXd & state) {
    Eigen::VectorXd prediction = F * state;
    prediction[3] = tools::limit_rad(prediction[3]);  // 偏航角限制在 [-pi, pi]
    return prediction;
  };

  // 调用基类的预测方法（EKF预测）
  tools::ExtendedKalmanFilter::predict(F, Q, transition);
}

/**
 * @brief 状态加法，并对偏航角归一化
 */
Eigen::VectorXd CAFromTJU::state_add(
  const Eigen::VectorXd & state, const Eigen::VectorXd & delta)
{
  Eigen::VectorXd result = state + delta;
  result[3] = tools::limit_rad(result[3]);
  return result;
}

}  // namespace auto_aim