#include "center_acceleration_estimator.hpp"

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace auto_aim
{
namespace
{

bool finite_nonnegative(double value) { return std::isfinite(value) && value >= 0.0; }

Eigen::Vector2d clamp_norm(const Eigen::Vector2d & value, double max_norm)
{
  const double norm = value.norm();
  if (norm <= max_norm || norm == 0.0) return value;
  return value * (max_norm / norm);
}

}  // namespace

CenterAccelerationEstimator::CenterAccelerationEstimator(CenterAccelerationEstimatorConfig config)
: config_(config)
{
  if (!std::isfinite(config_.window_seconds) || config_.window_seconds <= 0.0) {
    throw std::invalid_argument("center acceleration window must be positive");
  }
  if (config_.min_samples < 3) {
    throw std::invalid_argument("center acceleration fit requires at least three samples");
  }
  if (
    !std::isfinite(config_.min_span_seconds) || config_.min_span_seconds <= 0.0 ||
    config_.min_span_seconds > config_.window_seconds) {
    throw std::invalid_argument("center acceleration fit span must be within the window");
  }
  if (!std::isfinite(config_.ema_alpha) || config_.ema_alpha <= 0.0 || config_.ema_alpha > 1.0) {
    throw std::invalid_argument("center acceleration EMA alpha must be in (0, 1]");
  }
  if (!finite_nonnegative(config_.max_acceleration) || config_.max_acceleration == 0.0) {
    throw std::invalid_argument("center acceleration limit must be positive");
  }
  if (!finite_nonnegative(config_.max_jerk) || config_.max_jerk == 0.0) {
    throw std::invalid_argument("center acceleration jerk limit must be positive");
  }
  if (!finite_nonnegative(config_.max_fit_rmse)) {
    throw std::invalid_argument("center acceleration fit RMSE must be non-negative");
  }
  if (!std::isfinite(config_.stale_timeout_seconds) || config_.stale_timeout_seconds <= 0.0) {
    throw std::invalid_argument("center acceleration stale timeout must be positive");
  }
}

void CenterAccelerationEstimator::reset()
{
  samples_.clear();
  invalidate_estimate();
}

void CenterAccelerationEstimator::invalidate_estimate()
{
  acceleration_.setZero();
  estimate_valid_ = false;
  last_fit_rmse_ = std::numeric_limits<double>::infinity();
  has_last_fit_time_ = false;
}

void CenterAccelerationEstimator::add_sample(TimePoint time, const Eigen::Vector2d & position)
{
  if (!position.allFinite()) {
    reset();
    return;
  }
  if (!config_.enabled) return;

  if (!samples_.empty()) {
    const double gap = std::chrono::duration<double>(time - samples_.back().time).count();
    if (!std::isfinite(gap) || gap <= 0.0 || gap > config_.stale_timeout_seconds) reset();
  }

  samples_.push_back({time, position});
  while (samples_.size() > 1 &&
         std::chrono::duration<double>(time - samples_.front().time).count() >
           config_.window_seconds) {
    samples_.pop_front();
  }

  fit();
}

void CenterAccelerationEstimator::fit()
{
  if (samples_.size() < config_.min_samples) {
    invalidate_estimate();
    return;
  }

  const TimePoint reference_time = samples_.back().time;
  const double span = std::chrono::duration<double>(reference_time - samples_.front().time).count();
  if (!std::isfinite(span) || span < config_.min_span_seconds) {
    invalidate_estimate();
    return;
  }

  const Eigen::Index rows = static_cast<Eigen::Index>(samples_.size());
  Eigen::MatrixXd design(rows, 3);
  Eigen::MatrixXd positions(rows, 2);
  for (Eigen::Index row = 0; row < rows; ++row) {
    const auto & sample = samples_[static_cast<std::size_t>(row)];
    const double relative_time =
      std::chrono::duration<double>(sample.time - reference_time).count();
    design.row(row) << 1.0, relative_time, 0.5 * relative_time * relative_time;
    positions.row(row) = sample.position.transpose();
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(design);
  if (decomposition.rank() < 3) {
    invalidate_estimate();
    return;
  }

  const Eigen::Matrix<double, 3, 2> coefficients = decomposition.solve(positions);
  const Eigen::MatrixXd residual = positions - design * coefficients;
  const double rmse = std::sqrt(residual.squaredNorm() / static_cast<double>(2 * rows));
  const Eigen::Vector2d raw_acceleration = coefficients.row(2).transpose();
  if (
    !coefficients.allFinite() || !std::isfinite(rmse) || rmse > config_.max_fit_rmse ||
    !raw_acceleration.allFinite()) {
    invalidate_estimate();
    return;
  }

  const Eigen::Vector2d bounded_raw = clamp_norm(raw_acceleration, config_.max_acceleration);
  Eigen::Vector2d filtered =
    (1.0 - config_.ema_alpha) * acceleration_ + config_.ema_alpha * bounded_raw;

  double filter_dt = 0.0;
  if (has_last_fit_time_) {
    filter_dt = std::chrono::duration<double>(reference_time - last_fit_time_).count();
  } else if (samples_.size() >= 2) {
    filter_dt =
      std::chrono::duration<double>(samples_.back().time - samples_[samples_.size() - 2].time)
        .count();
  }
  if (!std::isfinite(filter_dt) || filter_dt <= 0.0) {
    invalidate_estimate();
    return;
  }

  const Eigen::Vector2d delta = filtered - acceleration_;
  const double max_delta = config_.max_jerk * filter_dt;
  if (delta.norm() > max_delta) filtered = acceleration_ + clamp_norm(delta, max_delta);

  acceleration_ = clamp_norm(filtered, config_.max_acceleration);
  estimate_valid_ = true;
  last_fit_rmse_ = rmse;
  last_fit_time_ = reference_time;
  has_last_fit_time_ = true;
}

bool CenterAccelerationEstimator::valid(TimePoint time) const
{
  if (!config_.enabled || !estimate_valid_ || samples_.empty()) return false;
  const double age = std::chrono::duration<double>(time - samples_.back().time).count();
  return std::isfinite(age) && age >= 0.0 && age <= config_.stale_timeout_seconds;
}

Eigen::Vector2d CenterAccelerationEstimator::acceleration(TimePoint time) const
{
  return valid(time) ? acceleration_ : Eigen::Vector2d::Zero();
}

}  // namespace auto_aim
