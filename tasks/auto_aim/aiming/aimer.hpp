#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/gimbal/cboard.hpp"
#include "io/gimbal/command.hpp"
#include "../tracking/target.hpp"

namespace auto_aim
{

struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  /** @brief 根据配置文件初始化瞄准参数 @param config_path YAML 配置文件路径 */
  explicit Aimer(const std::string & config_path);
  /** @brief 为普通机器人选择目标并生成控制命令 @param targets 候选跟踪目标 @param timestamp 当前图像时间戳 @param bullet_speed 弹速，单位 m/s @param to_now 是否将目标预测到当前时刻 @return 云台控制与开火命令 */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  /** @brief 按哨兵射击侧选择目标并生成控制命令 @param targets 候选跟踪目标 @param timestamp 当前图像时间戳 @param bullet_speed 弹速，单位 m/s @param shoot_mode 左、右或双侧射击模式 @param to_now 是否将目标预测到当前时刻 @return 云台控制与开火命令 */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;
  double comming_angle_;
  double leaving_angle_;
  double lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;

  /** @brief 从目标的各装甲板中选择瞄准点 @param target 跟踪目标 @return 瞄准点及有效标志 */
  AimPoint choose_aim_point(const Target & target);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
