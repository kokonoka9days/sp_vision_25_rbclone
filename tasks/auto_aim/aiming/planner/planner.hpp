#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <memory>
#include <optional>
#include <string>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"
#include "tools/logger.hpp"
#include "tools/yaw_delay_model.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

/** @brief 检查开火采样偏移是否在规划时域内 @param offset 相对时域中心的采样偏移 @return 偏移有效时返回 true */
inline bool valid_shoot_offset(int offset)
{
  return HALF_HORIZON + offset >= 0 && HALF_HORIZON + offset < HORIZON;
}

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control = false;
  bool fire = false;
  float target_yaw = 0;
  float target_pitch = 0;
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  float yaw_delay = 0;
  int yaw_delay_direction = 0;
  bool yaw_reversing = false;
};

struct TinySolverDeleter
{
  /** @brief 释放 TinyMPC 求解器 @param solver 求解器指针 */
  void operator()(TinySolver * solver) const noexcept { tiny_cleanup(solver); }
};

using TinySolverHandle = std::unique_ptr<TinySolver, TinySolverDeleter>;

class Planner
{
public:
  enum ShootStrategy{//开火策略
    Dynamics,          //动力学
    rbSuppressiveFire, //旧火控,火力压制
    rbHero,             //英雄
    SB                  //哨兵
  };
  Eigen::Vector4d debug_xyza;
  double aim_target_yaw;
  /** @brief 根据配置初始化轨迹规划器与 MPC 求解器 @param config_path YAML 配置路径 */
  Planner(const std::string & config_path);
  /** @brief 深拷贝规划器并重新创建求解器 @param other 源规划器 */
  Planner(const Planner & other);
  /** @brief 深拷贝赋值规划器 @param other 源规划器 @return 当前规划器 */
  Planner & operator=(const Planner & other);
  /** @brief 移动构造规划器 @param other 源规划器 */
  Planner(Planner && other) noexcept = default;
  /** @brief 移动赋值规划器 @param other 源规划器 @return 当前规划器 */
  Planner & operator=(Planner && other) noexcept = default;

  /** @brief 使用动力学策略规划云台轨迹 @param target 跟踪目标 @param bullet_speed 弹速，单位 m/s @return 云台控制计划 */
  Plan plan(Target target, double bullet_speed);
  /** @brief 预测目标并按指定策略规划 @param target 可选跟踪目标 @param bullet_speed 弹速，单位 m/s @param gimbal_yaw 当前云台偏航角 @param strategy 开火策略 @return 云台控制计划；无目标时 control 为 false */
  Plan plan(
    std::optional<Target> target, double bullet_speed, double gimbal_yaw = 0,
    ShootStrategy strategy = Dynamics);
  /** @brief 使用步兵压制射击策略规划 @param target 跟踪目标 @param bullet_speed 弹速 @param gimbal_yaw 当前云台偏航角 @return 云台控制计划 */
  Plan rbplan(Target target, double bullet_speed, double gimbal_yaw);
  /** @brief 使用哨兵策略规划 @param target 跟踪目标 @param bullet_speed 弹速 @param gimbal_yaw 当前云台偏航角 @return 云台控制计划 */
  Plan sbplan(Target target, double bullet_speed, double gimbal_yaw);
  /** @brief 判断步兵策略当前是否允许开火 @param target 跟踪目标 @param gimbal_yaw 当前云台偏航角 @param tower_fixed_pitch 是否使用前哨站固定俯仰约束 @return 允许开火时返回 true */
  bool rbShoot(Target target, double gimbal_yaw,  bool tower_fixed_pitch = false);
  /** @brief 使用英雄策略规划 @param target 跟踪目标 @param bullet_speed 弹速 @param gimbal_yaw 当前云台偏航角 @return 云台控制计划 */
  Plan rbHeroplan(Target target, double bullet_speed, double gimbal_yaw); 
private:
  bool is_far = false;
  bool is_high = false;
  double yaw_offset_;
  double pitch_offset_;
  double far_pitch_offset_;
  double far_high_pitch_offset_;
  double fire_thresh_;
  double target_dist_error_, target_h_error_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double small_armor_tolerance, big_armor_tolerance;
  double tower_and_base_armor_tolerance_;
  double gimbal_control_delay;
  double tower_pitch_prediction_time_;

  TinySolverHandle yaw_solver_;
  TinySolverHandle pitch_solver_;
  std::string config_path_;

  int last_selected_idx = -1;
  Eigen::Vector3d last_selected_xyz = Eigen::Vector3d::Zero();
  tools::YawDelayModel yaw_delay_model_;
  double last_yaw_command_velocity_ = 0;

  /** @brief 初始化偏航轴 MPC 求解器 @param config_path YAML 配置路径 */
  void setup_yaw_solver(const std::string & config_path);
  /** @brief 初始化俯仰轴 MPC 求解器 @param config_path YAML 配置路径 */
  void setup_pitch_solver(const std::string & config_path);

  /** @brief 计算动力学策略瞄准角 @param target 跟踪目标 @param bullet_speed 弹速 @return yaw、pitch */
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  /** @brief 计算步兵策略瞄准角 @param target 跟踪目标 @param bullet_speed 弹速 @return yaw、pitch */
  Eigen::Matrix<double, 2, 1> rbaim(const Target & target, double bullet_speed);
  /** @brief 计算英雄策略瞄准角 @param target 跟踪目标 @param bullet_speed 弹速 @param gimbal_yaw 当前云台偏航角 @return yaw、pitch */
  Eigen::Matrix<double, 2, 1> heroaim(const Target & target, double bullet_speed, double gimbal_yaw);
  /** @brief 生成动力学策略预测轨迹 @param target 跟踪目标 @param yaw0 初始偏航角 @param bullet_speed 弹速 @return 规划时域轨迹 */
  Trajectory get_trajectory(Target  target, double yaw0, double bullet_speed);
  /** @brief 生成步兵策略预测轨迹 @param target 跟踪目标 @param yaw0 初始偏航角 @param bullet_speed 弹速 @return 规划时域轨迹 */
  Trajectory rbget_trajectory(Target target, double yaw0, double bullet_speed);
  /** @brief 使用独立 yaw/pitch 目标预测生成步兵轨迹 */
  Trajectory rbget_trajectory_split(
    Target yaw_target, Target pitch_target, double yaw0, double bullet_speed);

  std::chrono::steady_clock::time_point outpost_z_stable_start_time_;
  bool outpost_is_make = true;

  double gimbal_delay_;

  int shoot_offset_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
