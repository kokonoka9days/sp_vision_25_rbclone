#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  void reset();

  const Target * predicted_target() const { return predicted_target_.get(); }

  double angle;      ///
  double t_gap = 0;  ///

private:
  SmallTarget target_;
  double yaw_offset_;
  double pitch_offset_;

  double fire_gap_time_;
  double small_predict_time_;
  double big_predict_time_;
  double max_yaw_vel_ = 4.0;
  double max_pitch_vel_ = 3.0;
  double max_yaw_acc_ = 25.0;
  double max_pitch_acc_ = 30.0;
  double command_position_gain_ = 18.0;
  double fire_angle_error_ = 1.0 / 57.3;

  bool solution_converged_ = false;
  std::unique_ptr<Target> predicted_target_;

  // for mpc
  bool command_state_initialized_ = false;
  double command_yaw_ = 0.0;
  double command_pitch_ = 0.0;
  double command_yaw_vel_ = 0.0;
  double command_pitch_vel_ = 0.0;
  std::chrono::steady_clock::time_point last_command_t_{};

  std::chrono::steady_clock::time_point last_fire_t_;

  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch, bool save_prediction = false);

  double predict_time(const Target & target) const;

  void update_command(
    double target_yaw, double target_pitch, double target_yaw_vel, double target_pitch_vel,
    const io::GimbalState & gs, std::chrono::steady_clock::time_point now, auto_aim::Plan & plan);
};
}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
