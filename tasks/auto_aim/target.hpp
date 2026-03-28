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
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  // IMM: 由于有两个EKF，我们对外隐藏EKF实例，直接提供最终融合的X和P(可选)
  const tools::ExtendedKalmanFilter & ekf() const; 
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  inline Eigen::VectorXd getEKFXest() {
    return combined_x_; // IMM: 返回融合后的状态
  }

  inline std::chrono::steady_clock::time_point getTimePoint() {
    return t_;
  }

  //前哨站
  double tower_armor_hs[3] = {0, 0, 0};  
  double tower_armor_h;
  double tower_armor_hs_datas[3] = {0, 0, 0}; 
  double last_tower_armor_h[3] = {0, 0, 0};
  int tower_armor_hs_datas_ptr = 0;

  //长短焦
  bool cam_is_short = true;
  bool last_cam_is_short = true;
  std::chrono::steady_clock::time_point cam_is_switch_time_point; //相机切换时间点；

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  // IMM (Interacting Multiple Model) 变量
  tools::ExtendedKalmanFilter ekf_1_; // 模型1: 低过程噪声 (平滑)
  tools::ExtendedKalmanFilter ekf_2_; // 模型2: 高过程噪声 (机动)
  Eigen::VectorXd combined_x_;        // 最终融合的状态向量
  Eigen::Vector2d mu_;                // 两个模型的当前概率
  Eigen::Matrix2d P_trans_;           // 马尔可夫状态转移概率矩阵

  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  // 用于在 IMM 混合状态时，安全处理角度 (Yaw) 的环绕问题
  Eigen::VectorXd mix_states(const Eigen::VectorXd& xa, const Eigen::VectorXd& xb, double wa, double wb) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP