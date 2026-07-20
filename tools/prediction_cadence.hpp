#ifndef TOOLS__PREDICTION_CADENCE_HPP
#define TOOLS__PREDICTION_CADENCE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>

namespace tools
{
// Converts the observed detector period into a control period with configurable prediction steps.
class PredictionCadence
{
public:
  explicit PredictionCadence(int prediction_frames_between_detections)
  : prediction_frames_between_detections_(prediction_frames_between_detections)
  {
  }

  void observe(std::chrono::steady_clock::time_point timestamp)
  {
    if (last_timestamp_ != std::chrono::steady_clock::time_point{} && timestamp > last_timestamp_) {
      const double sample_period = std::chrono::duration<double>(timestamp - last_timestamp_).count();
      if (std::isfinite(sample_period) && sample_period >= 0.005 && sample_period <= 0.2) {
        const double previous_period = observed_period_s_.load(std::memory_order_relaxed);
        constexpr double smoothing = 0.2;
        observed_period_s_.store(
          previous_period * (1.0 - smoothing) + sample_period * smoothing,
          std::memory_order_relaxed);
      }
    }
    last_timestamp_ = timestamp;
  }

  double control_period_s() const
  {
    const double divisor = static_cast<double>(prediction_frames_between_detections_) + 1.0;
    constexpr double minimum_control_period_s = 0.002;
    return std::max(
      minimum_control_period_s,
      observed_period_s_.load(std::memory_order_relaxed) / divisor);
  }

private:
  int prediction_frames_between_detections_;
  std::atomic<double> observed_period_s_{1.0 / 50.0};
  std::chrono::steady_clock::time_point last_timestamp_{};
};
}  // namespace tools

#endif  // TOOLS__PREDICTION_CADENCE_HPP
