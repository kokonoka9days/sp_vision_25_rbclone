#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  inline Eigen::VectorXd getEKFXest(){
    return ekf_.x;
  }

  inline std::chrono::steady_clock::time_point getTimePoint(){
    return t_;
  }

  // ==================== 前哨站专用接口 Start ====================
  inline bool isTower() const { return name == ArmorName::outpost; }
  inline bool isTowerInitialized() const { return tower_initialized_; }
  
  // 获取当前正在被击打的装甲板ID (0:低, 1:中, 2:高)
  inline int getCurrentTowerArmorId() const { return current_tower_armor_id_; }

  // 判断角速度方向 (EKF状态 x[7] 为角速度)
  // 返回 true 表示逆时针(CCW, w>0), false 表示顺时针(CW, w<0)
  bool getTowerVyawPositive() const {
    return ekf_.x[7] > 0;
  }
  
  // 根据旋转方向预测下一块出现的装甲板ID
  // 26赛季规则推导：
  // CCW (w>0): 观测顺序 0->2->1->0
  // CW  (w<0): 观测顺序 0->1->2->0
  int getNextTowerArmorId() const {
    return getTowerVyawPositive() ? 
        (current_tower_armor_id_ + 2) % 3 : (current_tower_armor_id_ + 1) % 3;
  }
  // ==================== 前哨站专用接口 End ====================

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  // ==================== 前哨站专用变量 Start ====================
  int current_tower_armor_id_ = 0;   // 当前装甲板编号 [0, 1, 2]
  bool tower_initialized_ = false;   // 是否已完成初始化（锁定ID序列）
  double last_tower_z_ = 0.0;        // 上一次观测的Z轴高度
  double now_tower_z_ = 0.0;         // 当前观测的Z轴高度
  
  // 更新前哨站高度观测并尝试解算ID
  void updateTowerInfo(const Armor & armor);
  // 前哨站逻辑解算核心
  void solveTowerLogic();
  // ==================== 前哨站专用变量 End ====================

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP