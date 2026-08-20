#include "yaw_delay_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "math_tools.hpp"

namespace tools
{
namespace
{
std::vector<double> unwrap(const std::vector<double> & angles)
{
  if (angles.empty()) return {};
  std::vector<double> result(angles.size());
  result[0] = angles[0];
  for (std::size_t i = 1; i < angles.size(); ++i) {
    result[i] = result[i - 1] + limit_rad(angles[i] - angles[i - 1]);
  }
  return result;
}

double percentile95(std::vector<double> values)
{
  if (values.empty()) return 0;
  const auto index = static_cast<std::size_t>(0.95 * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}
}  // namespace

YawDelayModel::YawDelayModel(
  std::vector<YawDelayPoint> positive, std::vector<YawDelayPoint> negative,
  double reverse_penalty_s, double direction_deadband_rad_s, double reverse_window_s)
: positive_(std::move(positive)),
  negative_(std::move(negative)),
  reverse_penalty_s_(reverse_penalty_s),
  direction_deadband_rad_s_(direction_deadband_rad_s),
  reverse_window_s_(reverse_window_s)
{
  validate_points(positive_, "positive");
  validate_points(negative_, "negative");
  if (!std::isfinite(reverse_penalty_s_) || reverse_penalty_s_ < 0 || reverse_penalty_s_ > 0.2) {
    throw std::invalid_argument("yaw reverse penalty must be in [0, 0.2] seconds");
  }
  if (!std::isfinite(direction_deadband_rad_s_) || direction_deadband_rad_s_ <= 0) {
    throw std::invalid_argument("yaw direction deadband must be positive");
  }
  if (!std::isfinite(reverse_window_s_) || reverse_window_s_ < 0 || reverse_window_s_ > 0.2) {
    throw std::invalid_argument("yaw reverse window must be in [0, 0.2] seconds");
  }
  const auto max_delay = [](const std::vector<YawDelayPoint> & points) {
    return std::max_element(
      points.begin(), points.end(),
      [](const auto & lhs, const auto & rhs) { return lhs.delay_s < rhs.delay_s; })
      ->delay_s;
  };
  if (reverse_penalty_s_ + std::max(max_delay(positive_), max_delay(negative_)) > 0.2) {
    throw std::invalid_argument("yaw curve delay plus reverse penalty must be in [0, 0.2] seconds");
  }
}

bool YawDelayModel::enabled() const noexcept
{
  return !positive_.empty() && !negative_.empty();
}

int YawDelayModel::direction(double command_velocity) const noexcept
{
  if (!std::isfinite(command_velocity) || std::abs(command_velocity) < direction_deadband_rad_s_) {
    return 0;
  }
  return command_velocity > 0 ? 1 : -1;
}

double YawDelayModel::query(
  double command_velocity, double previous_command_velocity,
  std::chrono::steady_clock::time_point now)
{
  if (!enabled() || !std::isfinite(command_velocity)) return 0;

  const int current_direction = direction(command_velocity);
  const int previous_direction = direction(previous_command_velocity);
  // Keep the last meaningful direction while inside the deadband. This makes
  // a command that briefly crosses zero use the same directional curve rather
  // than jumping to the positive curve because of measurement noise.
  const int curve_direction =
    current_direction != 0 ? current_direction
                           : (last_direction_ != 0 ? last_direction_ : previous_direction);

  // Arm the penalty once for each sign transition. `last_direction_` tracks
  // the last non-deadband command, so repeated frames during the window do not
  // accumulate the penalty.
  const bool sign_changed_from_state =
    last_direction_ != 0 && current_direction != last_direction_;
  const bool sign_changed_from_previous =
    last_direction_ == 0 && previous_direction != 0 && current_direction != previous_direction;
  if (
    current_direction != 0 && (sign_changed_from_state || sign_changed_from_previous) &&
    reverse_window_s_ > 0) {
    reverse_until_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<double>(reverse_window_s_));
  }
  if (current_direction != 0) last_direction_ = current_direction;

  const auto & curve = curve_direction < 0 ? negative_ : positive_;
  const double base_delay = interpolate(curve, std::abs(command_velocity));
  return base_delay + (reversal_active(now) ? reverse_penalty_s_ : 0.0);
}

bool YawDelayModel::reversal_active(std::chrono::steady_clock::time_point now) const noexcept
{
  return reverse_until_.time_since_epoch().count() != 0 && now < reverse_until_;
}

void YawDelayModel::reset() noexcept
{
  last_direction_ = 0;
  reverse_until_ = {};
}

void YawDelayModel::validate_points(const std::vector<YawDelayPoint> & points, const char * name)
{
  if (points.empty()) throw std::invalid_argument(std::string("yaw ") + name + " curve is empty");
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto & point = points[i];
    if (
      !std::isfinite(point.speed_rad_s) || !std::isfinite(point.delay_s) ||
      point.speed_rad_s < 0 || point.delay_s < 0 || point.delay_s > 0.2) {
      throw std::invalid_argument(std::string("yaw ") + name + " curve contains an invalid point");
    }
    if (i > 0 && point.speed_rad_s <= points[i - 1].speed_rad_s) {
      throw std::invalid_argument(std::string("yaw ") + name + " curve speeds must be increasing");
    }
  }
}

double YawDelayModel::interpolate(const std::vector<YawDelayPoint> & points, double speed)
{
  if (points.empty()) return 0;
  if (speed <= points.front().speed_rad_s) return points.front().delay_s;
  if (speed >= points.back().speed_rad_s) return points.back().delay_s;

  const auto upper = std::upper_bound(
    points.begin(), points.end(), speed,
    [](double value, const YawDelayPoint & point) { return value < point.speed_rad_s; });
  const auto lower = upper - 1;
  const double ratio =
    (speed - lower->speed_rad_s) / (upper->speed_rad_s - lower->speed_rad_s);
  return lower->delay_s + ratio * (upper->delay_s - lower->delay_s);
}

YawDelayEstimate estimate_yaw_delay(
  const std::vector<YawDelaySample> & samples, double sample_period_s, double max_delay_s,
  double min_correlation)
{
  YawDelayEstimate result;
  if (
    samples.size() < 20 || !std::isfinite(sample_period_s) || sample_period_s <= 0 ||
    !std::isfinite(max_delay_s) || max_delay_s <= 0 || !std::isfinite(min_correlation)) {
    result.reason = "not enough samples or invalid timing parameters";
    return result;
  }

  std::vector<YawDelaySample> sorted = samples;
  std::sort(sorted.begin(), sorted.end(), [](const auto & lhs, const auto & rhs) {
    return lhs.time_s < rhs.time_s;
  });
  if (!(sorted.back().time_s > sorted.front().time_s)) {
    result.reason = "continuous segment has no positive duration";
    return result;
  }

  std::vector<double> raw_command, raw_feedback;
  raw_command.reserve(sorted.size());
  raw_feedback.reserve(sorted.size());
  for (const auto & sample : sorted) {
    if (
      !std::isfinite(sample.time_s) || !std::isfinite(sample.command_rad) ||
      !std::isfinite(sample.feedback_rad)) {
      result.reason = "samples contain non-finite values";
      return result;
    }
    raw_command.push_back(sample.command_rad);
    raw_feedback.push_back(sample.feedback_rad);
  }
  raw_command = unwrap(raw_command);
  raw_feedback = unwrap(raw_feedback);

  std::vector<double> command, feedback;
  std::size_t source = 0;
  for (double time = sorted.front().time_s; time <= sorted.back().time_s;
       time += sample_period_s) {
    while (source + 1 < sorted.size() && sorted[source + 1].time_s < time) ++source;
    if (source + 1 >= sorted.size()) break;
    const double dt = sorted[source + 1].time_s - sorted[source].time_s;
    if (dt <= 0) continue;
    const double ratio = std::clamp(
      (time - sorted[source].time_s) / dt, 0.0, 1.0);
    command.push_back(raw_command[source] + ratio * (raw_command[source + 1] - raw_command[source]));
    feedback.push_back(
      raw_feedback[source] + ratio * (raw_feedback[source + 1] - raw_feedback[source]));
  }
  if (command.size() < 20) {
    result.reason = "resampled segment is too short";
    return result;
  }

  std::vector<double> command_velocity(command.size() - 1);
  std::vector<double> feedback_velocity(feedback.size() - 1);
  std::vector<double> absolute_speed;
  absolute_speed.reserve(command_velocity.size());
  for (std::size_t i = 1; i < command.size(); ++i) {
    command_velocity[i - 1] = (command[i] - command[i - 1]) / sample_period_s;
    feedback_velocity[i - 1] = (feedback[i] - feedback[i - 1]) / sample_period_s;
    absolute_speed.push_back(std::abs(command_velocity[i - 1]));
  }
  result.speed_rad_s = percentile95(std::move(absolute_speed));
  result.sample_count = command_velocity.size();

  const int max_lag = std::min<int>(
    static_cast<int>(max_delay_s / sample_period_s),
    static_cast<int>(command_velocity.size() / 4));
  if (max_lag < 1) {
    result.reason = "segment is too short for the requested delay range";
    return result;
  }
  std::vector<double> correlations(max_lag + 1, -1.0);
  for (int lag = 0; lag <= max_lag; ++lag) {
    const std::size_t count = command_velocity.size() - static_cast<std::size_t>(lag);
    double command_mean = 0;
    double feedback_mean = 0;
    for (std::size_t i = 0; i < count; ++i) {
      command_mean += command_velocity[i];
      feedback_mean += feedback_velocity[i + lag];
    }
    command_mean /= static_cast<double>(count);
    feedback_mean /= static_cast<double>(count);

    double numerator = 0;
    double command_energy = 0;
    double feedback_energy = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const double lhs = command_velocity[i] - command_mean;
      const double rhs = feedback_velocity[i + lag] - feedback_mean;
      numerator += lhs * rhs;
      command_energy += lhs * lhs;
      feedback_energy += rhs * rhs;
    }
    const double denominator = std::sqrt(command_energy * feedback_energy);
    if (denominator > std::numeric_limits<double>::epsilon()) {
      correlations[lag] = numerator / denominator;
    }
  }

  const auto peak = std::max_element(correlations.begin(), correlations.end());
  const double peak_value = *peak;
  // Periodic motion can produce several nearly identical correlation peaks.
  // Prefer the earliest peak within the numerical tolerance so the estimator
  // reports the smallest physically plausible command-to-feedback delay.
  const double peak_tolerance = 1e-6;
  const auto first_peak = std::find_if(
    correlations.begin(), correlations.end(),
    [peak_value, peak_tolerance](double value) { return value >= peak_value - peak_tolerance; });
  const int peak_index = static_cast<int>(std::distance(correlations.begin(), first_peak));
  double refined_index = peak_index;
  if (peak_index > 0 && peak_index < max_lag) {
    const double left = correlations[peak_index - 1];
    const double center = correlations[peak_index];
    const double right = correlations[peak_index + 1];
    const double denominator = left - 2.0 * center + right;
    if (std::abs(denominator) > 1e-12) {
      refined_index += 0.5 * (left - right) / denominator;
    }
  }
  result.delay_s = std::max(0.0, refined_index * sample_period_s);
  result.correlation = correlations[peak_index];
  result.valid = result.correlation >= min_correlation;
  if (!result.valid) result.reason = "correlation peak is below threshold";
  return result;
}
}  // namespace tools
