#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "io/gimbal/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "aimer.hpp"

namespace auto_aim
{
class Shooter
{
public:
  /** @brief 根据配置初始化开火判定器 @param config_path YAML 配置文件路径 */
  Shooter(const std::string & config_path);

  /** @brief 根据普通控制命令和目标状态判定是否开火 @param command 当前控制命令 @param aimer 瞄准器 @param targets 跟踪目标列表 @param gimbal_pos 云台当前位置 @return 满足开火条件时返回 true */
  bool shoot(
    const io::Command & command, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

  /** @brief 根据哨兵控制帧和目标状态判定是否开火 @param vision_cmd 哨兵控制帧 @param aimer 瞄准器 @param targets 跟踪目标列表 @param gimbal_pos 云台当前位置 @return 满足开火条件时返回 true */
  bool shoot_g(
    const io::sb_VisionToGimbal & vision_cmd, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
  io::Command last_command_;
  double last_yaw_;
  double judge_distance_;
  double first_tolerance_;
  double second_tolerance_;
  bool auto_fire_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__SHOOTER_HPP
