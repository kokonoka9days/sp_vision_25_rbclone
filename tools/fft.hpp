#ifndef PERIODIC_MOTION_ANALYZER_HPP
#define PERIODIC_MOTION_ANALYZER_HPP

#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <optional>

#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

namespace tools
{

class FFTExample
{
public:
  using Buffer = boost::circular_buffer<double>;
  using TimePoint = std::chrono::steady_clock::time_point;

  FFTExample(std::size_t max_points = 400)
  : t_buf_(max_points), val_buf_(max_points)
  {
  }

  void add_sample(double t, double val);
  void add_sample(TimePoint t, double val);

  double low_pass(double value, double alpha = 0.2);
  void reset_low_pass();

  bool analyze(bool force = false);

  double get_acceleration(double t) const;
  double get_value(double t) const;
  double get_value(TimePoint t) const;
  double get_val_buf_front() const;

  bool get_is_periodic() const;
  void set_phase_offset(double offset);
  double get_mean_val() const;

private:
  Buffer t_buf_, val_buf_;
  mutable std::mutex mtx_;
  bool is_periodic_ = false;
  double last_freq_ = 0.0;
  double last_amp_ = 0.0;
  double last_phase_ = 0.0;
  double last_period_ = 0.0;
  double phase_offset_ = 0.0;
  std::size_t min_points_ = 400;
  std::size_t analysis_interval_frames_ = 30;
  std::size_t last_analysis_frames_ = 0;
  double mean_val = 0.0;
  bool low_pass_initialized_ = false;
  double low_pass_value_ = 0.0;
  std::optional<TimePoint> time_origin_;

  double detect_period_by_autocorr(const Eigen::VectorXd & x, double dt);
  bool extract_base_harmonic(
    const Eigen::VectorXd & x, double dt, double & freq, double & amp, double & phase);
};

}  // namespace tools

#endif  // PERIODIC_MOTION_ANALYZER_HPP
