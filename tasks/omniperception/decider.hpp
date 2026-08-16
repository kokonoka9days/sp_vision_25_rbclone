#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <iostream>
#include <list>
#include <unordered_map>

// #include "armor.hpp"
#include "detection.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/solver.hpp"

namespace auto_aim
{
class YOLO;
}

namespace omniperception
{
class Decider
{
public:
  /** @brief 根据配置初始化全向感知决策器 @param config_path YAML 配置路径 */
  Decider(const std::string & config_path);

  /** @brief 融合左右全向相机并生成云台帧 @param yolo 检测器 @param gimbal_pos 云台位置 @param omn_cam1_l 左相机 @param omn_cam2_r 右相机 @param left_solver 左相机求解器 @param right_solver 右相机求解器 @param target_distance 可选输出目标距离 @return 云台控制帧 */
  io::VisionToGimbal decide_g(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & omn_cam1_l,
    io::Camera & omn_cam2_r, const auto_aim::Solver & left_solver,
    const auto_aim::Solver & right_solver, float * target_distance = nullptr);
  
  /** @brief 融合两路 USB 相机和后相机生成控制命令 @param yolo 检测器 @param gimbal_pos 云台位置 @param usbcam1 USB 相机一 @param usbcam2 USB 相机二 @param back_camera 后相机 @return 控制命令 */
  io::Command decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
  io::USBCamera & usbcam2, io::Camera & back_camera);

  /** @brief 使用单路后相机生成控制命令 @param yolo 检测器 @param gimbal_pos 云台位置 @param back_cammera 后相机 @return 控制命令 */
  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera);

  /** @brief 根据已有检测结果生成控制命令 @param detection_queue 多相机检测结果 @return 控制命令 */
  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  /** @brief 计算二维装甲板相对光轴角度 @param armors 装甲板列表 @param camera 相机标识 @return 偏航和俯仰增量 */
  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  /** @brief 基于三维求解计算相对角度 @param armors 装甲板列表 @param camera 相机标识 @param left_solver 左求解器 @param right_solver 右求解器 @return 偏航和俯仰增量 */
  Eigen::Vector2d delta_angle_3d(
     std::list<auto_aim::Armor> & armors, const std::string & camera, 
            const auto_aim::Solver & left_solver, const auto_aim::Solver & right_solver  );

  /** @brief 过滤不可用于全向决策的装甲板 @param armors 待原地过滤列表 @return 过滤后非空时返回 true */
  bool armor_filter(std::list<auto_aim::Armor> & armors);

  /** @brief 过滤基地装甲板 @param armors 待原地过滤列表 @return 过滤后非空时返回 true */
  bool not_base_armor_filter(std::list<auto_aim::Armor> & armors);

  /** @brief 根据当前模式设置装甲板优先级 @param armors 装甲板列表 */
  void set_priority(std::list<auto_aim::Armor> & armors);
  /** @brief 过滤每路检测结果并按优先级排序 @param detection_queue 检测结果队列 */
  void sort(std::vector<DetectionResult> & detection_queue);

  /** @brief 汇总目标位置和类别信息 @param armors 装甲板列表 @param targets 跟踪目标列表 @return 目标四维信息 */
  Eigen::Vector4d get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

  /** @brief 更新处于无敌状态的敌方编号 @param invincible_enemy_ids 敌方编号列表 */
  void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

  /** @brief 按导航指定目标过滤装甲板 @param armors 待原地过滤列表 @param auto_aim_target 导航目标编号 */
  void get_auto_aim_target(
    std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target);

  io::Gimbal* gimbal_ = nullptr; // 新增一个云台指针，默认为空

  /** @brief 绑定云台状态源 @param gimbal 非拥有云台指针 */
  void set_gimbal(io::Gimbal* gimbal) { gimbal_ = gimbal; }

  /** @brief 设置优先级模式 @param mode 模式编号 */
  void set_mode(int mode){
    this->mode_ = mode;
  }

private:
  int img_width_;
  int img_height_;
  double fov_h_, new_fov_h_;
  double fov_v_, new_fov_v_;
  int mode_;
  int count_;
  int last_count_ = -1;
  io::VisionToGimbal last_vision_cmd; 

  auto_aim::Color enemy_color_;
  std::string enemy_color_str_;
  std::vector<auto_aim::ArmorName> invincible_armor_;  //无敌状态机器人编号,英雄为1，哨兵为6

  // 定义ArmorName到ArmorPriority的映射类型
  using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

  const PriorityMap mode1 = {
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::forth},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::fifth}};

  const PriorityMap mode2 = {
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};

  const PriorityMap mode3 = {
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};
};

enum PriorityMode
{
  MODE_ONE = 1,
  MODE_TWO,
  MODE_THREE
};

}  // namespace omniperception

#endif
