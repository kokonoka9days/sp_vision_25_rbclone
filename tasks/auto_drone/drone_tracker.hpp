#ifndef AUTO_DRONE__TRACKER_HPP
#define AUTO_DRONE__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <string>

#include "io/gimbal/gimbal.hpp"
#include "drone_armor.hpp"
#include "drone_solver.hpp"
#include "drone_target.hpp"

namespace auto_drone
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver * solver);

  std::string state() const;

  // 核心追踪接口：输入当前帧识别到的无人机列表
  std::vector<Target> track(
    std::vector<Drone> & drones, 
    std::chrono::steady_clock::time_point t);

  void set_gimbal(io::Gimbal* gimbal) { gimbal_ = gimbal; }

private:
  Solver * solver_;
  io::Gimbal* gimbal_ = nullptr;

  Color enemy_color_;
  std::string enemy_color_str_;

  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;

  std::string state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;

  void state_machine(bool found);

  bool set_target(std::vector<Drone> & drones, std::chrono::steady_clock::time_point t);

  bool update_target(std::vector<Drone> & drones, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__TRACKER_HPP