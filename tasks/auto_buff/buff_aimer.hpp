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
#include "buff_config.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
class Aimer
{
public:
  /** @brief 从配置文件初始化能量机关瞄准器 @param config_path YAML 配置路径 */
  Aimer(const std::string & config_path);
  /** @brief 使用已解析配置初始化瞄准器 @param config_path YAML 配置路径 @param config 能量机关配置 */
  Aimer(const std::string & config_path, BuffConfig config);

  /** @brief 计算能量机关传统控制命令 @param target 目标状态，会按预测时间推进 @param timestamp 图像时间戳 @param bullet_speed 弹速，单位 m/s @param to_now 是否补偿到当前时刻 @return 控制和开火命令 */
  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  /** @brief 计算能量机关 MPC 控制计划 @param target 目标状态 @param timestamp 图像时间戳 @param gs 当前云台状态 @param to_now 是否补偿到当前时刻 @return 云台控制计划 */
  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  /** @brief 获取最近一次成功生成的预测目标 @return 预测目标指针；无有效解时为空 */
  const Target * predicted_target() const { return predicted_target_.get(); }

  double angle;      ///
  double t_gap = 0;  ///

private:
  const BuffConfig config_;
  double yaw_offset_;
  double pitch_offset_;

  double fire_gap_time_;
  double predict_time_;

  bool solution_converged_ = false;
  std::unique_ptr<Target> predicted_target_;

  // for mpc
  bool first_in_aimer_ = true;

  std::chrono::steady_clock::time_point last_fire_t_;

  /** @brief 计算预测时刻的发送角度 @param target 目标状态 @param predict_time 预测时长，单位 s @param bullet_speed 弹速 @param to_now 是否补偿到当前时刻 @param yaw 输出偏航角 @param pitch 输出俯仰角 @param save_prediction 是否保存预测目标快照 @return 求解成功时返回 true */
  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch, bool save_prediction = false);
};
}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
