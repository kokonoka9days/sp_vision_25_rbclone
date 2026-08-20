#ifndef TOOLS__YAW_DELAY_MODEL_HPP
#define TOOLS__YAW_DELAY_MODEL_HPP

#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tools
{
struct YawDelayPoint
{
  double speed_rad_s = 0;
  double delay_s = 0;
};

struct YawDelayEstimate
{
  bool valid = false;
  double delay_s = 0;
  double correlation = 0;
  double speed_rad_s = 0;
  std::size_t sample_count = 0;
  std::string reason;
};

/** @brief Speed- and direction-dependent yaw command delay model. */
class YawDelayModel
{
public:
  YawDelayModel() = default;
  YawDelayModel(
    std::vector<YawDelayPoint> positive, std::vector<YawDelayPoint> negative,
    double reverse_penalty_s, double direction_deadband_rad_s, double reverse_window_s);

  bool enabled() const noexcept;

  /**
   * @brief Query delay for a command velocity and arm a short reversal penalty when needed.
   * @param command_velocity Current requested yaw velocity, rad/s.
   * @param previous_command_velocity Last emitted yaw velocity, rad/s.
   * @param now Monotonic timestamp used for the reversal window.
   */
  double query(
    double command_velocity, double previous_command_velocity,
    std::chrono::steady_clock::time_point now);

  int direction(double command_velocity) const noexcept;
  bool reversal_active(std::chrono::steady_clock::time_point now) const noexcept;
  void reset() noexcept;

private:
  static void validate_points(const std::vector<YawDelayPoint> & points, const char * name);
  static double interpolate(const std::vector<YawDelayPoint> & points, double speed);

  std::vector<YawDelayPoint> positive_;
  std::vector<YawDelayPoint> negative_;
  double reverse_penalty_s_ = 0;
  double direction_deadband_rad_s_ = 0.2;
  double reverse_window_s_ = 0.05;
  int last_direction_ = 0;
  std::chrono::steady_clock::time_point reverse_until_{};
};

/** @brief A timestamped yaw command/feedback sample used by the offline analyzer. */
struct YawDelaySample
{
  double time_s = 0;
  double command_rad = 0;
  double feedback_rad = 0;
};

/** @brief Estimate a delay from one continuous yaw command/feedback segment. */
YawDelayEstimate estimate_yaw_delay(
  const std::vector<YawDelaySample> & samples, double sample_period_s = 0.005,
  double max_delay_s = 0.2, double min_correlation = 0.7);

}  // namespace tools

#endif  // TOOLS__YAW_DELAY_MODEL_HPP
