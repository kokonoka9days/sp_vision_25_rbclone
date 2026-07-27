#include "ca_from_tju.hpp"

#include <cmath>
#include <stdexcept>

#include "tools/math_tools.hpp"

namespace auto_aim
{

CAFromTJU::CAFromTJU(const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0)
: State2Est(x0, P0, state_add)
{
  if (
    x0.size() != kStateDimension || P0.rows() != kStateDimension ||
    P0.cols() != kStateDimension) {
    throw std::invalid_argument("CAFromTJU requires an 11-dimensional state");
  }
}

void CAFromTJU::kf_predict(
  double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises)
{
  // TrackQueueV3 is an autonomous model; acceleration is part of the state.
  (void)u;
  predict_model(dt, noises);
}

void CAFromTJU::mpc_predict(
  double dt, const Eigen::VectorXd & u, const Eigen::VectorXd noises)
{
  kf_predict(dt, u, noises);
}

void CAFromTJU::predict_model(
  double dt, const Eigen::VectorXd & process_noise_diagonal)
{
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

  F(0, 4) = dt;
  F(0, 8) = half_dt_squared;
  F(1, 5) = dt;
  F(1, 9) = half_dt_squared;
  F(2, 6) = dt;
  F(3, 7) = dt;
  F(3, 10) = half_dt_squared;
  F(4, 8) = dt;
  F(5, 9) = dt;
  F(7, 10) = dt;

  const Eigen::MatrixXd Q = process_noise_diagonal.asDiagonal();
  auto transition = [&F](const Eigen::VectorXd & state) {
    Eigen::VectorXd prediction = F * state;
    prediction[3] = tools::limit_rad(prediction[3]);
    return prediction;
  };

  tools::ExtendedKalmanFilter::predict(F, Q, transition);
}

Eigen::VectorXd CAFromTJU::state_add(
  const Eigen::VectorXd & state, const Eigen::VectorXd & delta)
{
  Eigen::VectorXd result = state + delta;
  result[3] = tools::limit_rad(result[3]);
  return result;
}

}  // namespace auto_aim
