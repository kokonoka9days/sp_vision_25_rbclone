#ifndef TOOLS__ERROR_STATE_EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__ERROR_STATE_EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <ceres/jet.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tools
{

struct EstimatorDiagnostics
{
  double residual_norm = 0;
  double nis = 0;
  int observation_dim = 0;
  double normalized_nis = 0;
};

template <int N_X, class PredictFunc>
class ErrorStateExtendedKalmanFilter
{
public:
  using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
  using VectorX = Eigen::Matrix<double, N_X, 1>;
  using Jet = ceres::Jet<double, N_X>;
  using JetVectorX = Eigen::Matrix<Jet, N_X, 1>;
  using InjectFunc = std::function<void(const VectorX &, VectorX &)>;
  using BoxMinusFunc = std::function<void(const VectorX &, const VectorX &, VectorX &)>;
  using InjectJetFunc = std::function<void(const JetVectorX &, JetVectorX &)>;
  using BoxMinusJetFunc =
    std::function<void(const JetVectorX &, const JetVectorX &, JetVectorX &)>;

  ErrorStateExtendedKalmanFilter() = default;

  template <class Inject, class BoxMinus>
  ErrorStateExtendedKalmanFilter(
    const Inject & inject, const BoxMinus & box_minus, const MatrixXX & initial_covariance)
  : inject_(inject),
    box_minus_(box_minus),
    inject_jet_(inject),
    box_minus_jet_(box_minus),
    covariance_(initial_covariance)
  {
  }

  void set_state(const VectorX & state) { state_ = state; }
  void set_covariance(const MatrixXX & covariance) { covariance_ = covariance; }
  const VectorX & state() const { return state_; }
  const MatrixXX & covariance() const { return covariance_; }
  const EstimatorDiagnostics & diagnostics() const { return diagnostics_; }
  void set_iteration_num(int n) { iteration_num_ = std::max(1, n); }

  bool predict(const PredictFunc & predict_func, const MatrixXX & process_noise)
  {
    diagnostics_ = {};
    const VectorX previous = state_;
    VectorX predicted;
    predict_func(previous.data(), predicted.data());
    if (!predicted.allFinite() || !process_noise.allFinite()) return false;

    JetVectorX previous_jet;
    JetVectorX predicted_jet;
    JetVectorX delta_jet;
    for (int i = 0; i < N_X; ++i) {
      previous_jet[i] = Jet(previous[i]);
      predicted_jet[i] = Jet(predicted[i]);
      delta_jet[i] = Jet(0);
      delta_jet[i].v[i] = 1;
    }

    JetVectorX perturbed_jet = previous_jet;
    inject_jet_(delta_jet, perturbed_jet);
    JetVectorX perturbed_predicted_jet;
    predict_func(perturbed_jet.data(), perturbed_predicted_jet.data());
    JetVectorX predicted_error_jet;
    box_minus_jet_(predicted_jet, perturbed_predicted_jet, predicted_error_jet);

    MatrixXX F;
    for (int i = 0; i < N_X; ++i) F.row(i) = predicted_error_jet[i].v.transpose();
    MatrixXX predicted_covariance = F * covariance_ * F.transpose() + process_noise;
    predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.transpose());
    if (!F.allFinite() || !predicted_covariance.allFinite()) return false;

    state_ = predicted;
    covariance_ = predicted_covariance;
    return true;
  }

  struct ObservationBase
  {
    virtual ~ObservationBase() = default;
    virtual int dim() const = 0;
    virtual void predict(const VectorX & state, Eigen::VectorXd & predicted) const = 0;
    virtual void residual_and_noise(
      const Eigen::VectorXd & predicted, Eigen::VectorXd & residual, Eigen::MatrixXd & noise) const = 0;
  };

  template <int N_Z, class MeasureFunc, class NoiseFunc, class ResidualFunc>
  struct Observation : ObservationBase
  {
    using VectorZ = Eigen::Matrix<double, N_Z, 1>;

    Observation(
      const VectorZ & observation, MeasureFunc measure, NoiseFunc noise, ResidualFunc residual)
    : observation_(observation),
      measure_(std::move(measure)),
      noise_(std::move(noise)),
      residual_(std::move(residual))
    {
    }

    int dim() const override { return N_Z; }

    void predict(const VectorX & state, Eigen::VectorXd & predicted) const override
    {
      VectorZ fixed;
      measure_(state.data(), fixed.data());
      predicted = fixed;
    }

    void residual_and_noise(
      const Eigen::VectorXd & predicted, Eigen::VectorXd & residual,
      Eigen::MatrixXd & noise) const override
    {
      VectorZ fixed_predicted;
      for (int i = 0; i < N_Z; ++i) fixed_predicted[i] = predicted[i];
      residual = residual_(fixed_predicted, observation_);
      noise = noise_(observation_);
    }

    VectorZ observation_;
    MeasureFunc measure_;
    NoiseFunc noise_;
    ResidualFunc residual_;
  };

  template <int N_Z, class MeasureFunc, class NoiseFunc, class ResidualFunc>
  std::shared_ptr<ObservationBase> make_observation(
    const Eigen::Matrix<double, N_Z, 1> & observation, MeasureFunc && measure,
    NoiseFunc && noise, ResidualFunc && residual)
  {
    using ObservationType = Observation<
      N_Z, std::decay_t<MeasureFunc>, std::decay_t<NoiseFunc>, std::decay_t<ResidualFunc>>;
    return std::make_shared<ObservationType>(
      observation, std::forward<MeasureFunc>(measure), std::forward<NoiseFunc>(noise),
      std::forward<ResidualFunc>(residual));
  }

  bool update_multi(const std::vector<std::shared_ptr<ObservationBase>> & observations)
  {
    int total_dim = 0;
    for (const auto & observation : observations) total_dim += observation->dim();
    diagnostics_ = {};
    if (total_dim == 0) return false;

    const VectorX prior_state = state_;
    const MatrixXX prior_covariance = covariance_;
    VectorX delta = VectorX::Zero();
    Eigen::MatrixXd H(total_dim, N_X);
    Eigen::VectorXd residual(total_dim);
    Eigen::MatrixXd R(total_dim, total_dim);
    Eigen::MatrixXd K(N_X, total_dim);

    for (int iteration = 0; iteration < iteration_num_; ++iteration) {
      VectorX evaluation_state = prior_state;
      inject_(delta, evaluation_state);
      R.setZero();
      int offset = 0;
      for (const auto & observation : observations) {
        Eigen::VectorXd predicted;
        Eigen::VectorXd observation_residual;
        Eigen::MatrixXd observation_noise;
        observation->predict(evaluation_state, predicted);
        observation->residual_and_noise(predicted, observation_residual, observation_noise);

        const int dimension = observation->dim();
        Eigen::MatrixXd observation_H(dimension, N_X);
        constexpr double epsilon = 1e-6;
        for (int i = 0; i < N_X; ++i) {
          VectorX plus_delta = delta;
          VectorX minus_delta = delta;
          plus_delta[i] += epsilon;
          minus_delta[i] -= epsilon;
          VectorX plus_state = prior_state;
          VectorX minus_state = prior_state;
          inject_(plus_delta, plus_state);
          inject_(minus_delta, minus_state);

          Eigen::VectorXd plus_predicted;
          Eigen::VectorXd minus_predicted;
          Eigen::VectorXd plus_residual;
          Eigen::VectorXd minus_residual;
          Eigen::MatrixXd unused_noise;
          observation->predict(plus_state, plus_predicted);
          observation->predict(minus_state, minus_predicted);
          observation->residual_and_noise(plus_predicted, plus_residual, unused_noise);
          observation->residual_and_noise(minus_predicted, minus_residual, unused_noise);
          observation_H.col(i) = -(plus_residual - minus_residual) / (2 * epsilon);
        }

        H.block(offset, 0, dimension, N_X) = observation_H;
        residual.segment(offset, dimension) = observation_residual;
        R.block(offset, offset, dimension, dimension) = observation_noise;
        offset += dimension;
      }

      if (!H.allFinite() || !residual.allFinite() || !R.allFinite()) {
        diagnostics_ = {};
        return false;
      }
      Eigen::MatrixXd innovation_covariance = H * prior_covariance * H.transpose() + R;
      Eigen::LDLT<Eigen::MatrixXd> ldlt(innovation_covariance);
      if (
        ldlt.info() != Eigen::Success || !ldlt.vectorD().allFinite() ||
        ldlt.vectorD().minCoeff() <= 0) {
        innovation_covariance.diagonal().array() += 1e-9;
        ldlt.compute(innovation_covariance);
      }
      if (
        ldlt.info() != Eigen::Success || !ldlt.vectorD().allFinite() ||
        ldlt.vectorD().minCoeff() <= 0) {
        diagnostics_ = {};
        return false;
      }

      const Eigen::MatrixXd PHt = prior_covariance * H.transpose();
      K = ldlt.solve(PHt.transpose()).transpose();
      const VectorX step = K * residual;
      if (!K.allFinite() || !step.allFinite()) {
        diagnostics_ = {};
        return false;
      }

      if (iteration == 0) {
        const Eigen::VectorXd whitened = ldlt.solve(residual);
        const double nis = residual.dot(whitened);
        diagnostics_.residual_norm = residual.norm();
        diagnostics_.nis = std::max(0.0, nis);
        diagnostics_.observation_dim = total_dim;
        diagnostics_.normalized_nis = diagnostics_.nis / total_dim;
      }
      delta += step;
      if (step.norm() < 1e-7) break;
    }

    VectorX posterior_state = prior_state;
    inject_(delta, posterior_state);
    const MatrixXX identity = MatrixXX::Identity();
    MatrixXX posterior_covariance =
      (identity - K * H) * prior_covariance * (identity - K * H).transpose() + K * R * K.transpose();

    MatrixXX reset_jacobian;
    constexpr double reset_epsilon = 1e-6;
    for (int i = 0; i < N_X; ++i) {
      VectorX plus_delta = delta;
      VectorX minus_delta = delta;
      plus_delta[i] += reset_epsilon;
      minus_delta[i] -= reset_epsilon;
      VectorX plus_state = prior_state;
      VectorX minus_state = prior_state;
      inject_(plus_delta, plus_state);
      inject_(minus_delta, minus_state);
      VectorX plus_error;
      VectorX minus_error;
      box_minus_(posterior_state, plus_state, plus_error);
      box_minus_(posterior_state, minus_state, minus_error);
      reset_jacobian.col(i) = (plus_error - minus_error) / (2 * reset_epsilon);
    }
    posterior_covariance = reset_jacobian * posterior_covariance * reset_jacobian.transpose();
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.transpose());
    if (!posterior_state.allFinite() || !posterior_covariance.allFinite()) {
      diagnostics_ = {};
      return false;
    }

    state_ = posterior_state;
    covariance_ = posterior_covariance;
    return true;
  }

private:
  InjectFunc inject_;
  BoxMinusFunc box_minus_;
  InjectJetFunc inject_jet_;
  BoxMinusJetFunc box_minus_jet_;
  VectorX state_ = VectorX::Zero();
  MatrixXX covariance_ = MatrixXX::Identity();
  int iteration_num_ = 1;
  EstimatorDiagnostics diagnostics_;
};

}  // namespace tools

#endif  // TOOLS__ERROR_STATE_EXTENDED_KALMAN_FILTER_HPP
