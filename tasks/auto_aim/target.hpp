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
  Eigen::Vector3d xyz_in_world;

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const std::vector<Armor> & armors);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const; 
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  inline Eigen::VectorXd getEKFXest() {
    return ekf_.x;
  }

  inline std::chrono::steady_clock::time_point getTimePoint() {
    return t_;
  }

  //前哨站
  double tower_armor_hs[3] = {0, 0, 0};  
  double tower_armor_h;
  double tower_armor_hs_datas[3] = {0,0,0}; 
  double last_tower_armor_h[3] = {0,0,0};
  int tower_armor_hs_datas_ptr[3] = {0, 0, 0};

  //长短焦
  bool cam_is_short = true;
  bool last_cam_is_short = true;
  std::chrono::steady_clock::time_point cam_is_switch_time_point; //相机切换时间点；

private:
  int armor_num_;
  int switch_count_;
  int update_count_;
  int reject_count_; // [新增] 马氏距离连续拒收计数器，用于防死锁

  bool is_switch_, is_converged_;

  // 单一 EKF 实例
  tools::ExtendedKalmanFilter ekf_; 

  std::chrono::steady_clock::time_point t_;

  enum class MotionState {
    TRANSLATION,          // 纯平移
    IN_PLACE_ROTATION,    // 原地旋转
    TRANSLATION_ROTATION  // 平移旋转(走位小陀螺)
  };

  MotionState motion_state_ = MotionState::TRANSLATION;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP