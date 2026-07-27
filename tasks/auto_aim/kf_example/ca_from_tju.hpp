#ifndef AUTO_AIM__KF_EXAMPLE__CA_FROM_TJU_HPP
#define AUTO_AIM__KF_EXAMPLE__CA_FROM_TJU_HPP

#include "state2est.hpp"

namespace auto_aim
{

/**
 * @brief Constant-acceleration state estimator based on TJU TrackQueueV3.
 *
 * State: [x, y, z, yaw, vx, vy, vz, vyaw, ax, ay, ayaw]
 * Measurement expected by the original model: [x, y, z, yaw]
 */
class CAFromTJU : public State2Est
{
public:
  static constexpr Eigen::Index kStateDimension = 11;

  CAFromTJU() = default;

  /**
   * @param x0 Initial 11-dimensional state.
   * @param P0 Initial 11x11 covariance.
   * @throws std::invalid_argument if either argument has the wrong dimensions.
   */
  CAFromTJU(const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0);

  /**
   * @brief Run one EKF prediction.
   *
   * The TJU model is autonomous, so @p u is intentionally unused. @p noises
   * contains the 11 diagonal entries of the process-noise covariance matrix.
   */
  void kf_predict(
    double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises) override;

  void mpc_predict(
    double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises) override;

  /**
   * @brief Apply the constant-acceleration transition and covariance prediction.
   * @param dt Prediction interval in seconds.
   * @param process_noise_diagonal Eleven non-negative diagonal entries of Q.
   */
  void predict_model(double dt, const Eigen::VectorXd & process_noise_diagonal);

private:
  static Eigen::VectorXd state_add(
    const Eigen::VectorXd & state, const Eigen::VectorXd & delta);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__KF_EXAMPLE__CA_FROM_TJU_HPP
