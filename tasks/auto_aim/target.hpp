#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>
#include <algorithm> // 包含 max, min

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

// 【新增】单板CA滤波器结构体 (9维状态: x, y, z, vx, vy, vz, ax, ay, az)
struct SingleArmorCA
{
  tools::ExtendedKalmanFilter kf;
  bool is_initialized = false;
  int update_count = 0;

  SingleArmorCA() = default;

  void init(const Eigen::Vector3d & xyz) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(9);
    x0.head(3) = xyz;
    Eigen::VectorXd P0_diag(9);
    P0_diag << 1, 1, 1, 10, 10, 10, 100, 100, 100;
    Eigen::MatrixXd P0 = P0_diag.asDiagonal();

    auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd { return a + b; };
    kf = tools::ExtendedKalmanFilter(x0, P0, x_add);
    is_initialized = true;
    update_count = 1;
  }

  void predict(double dt) {
    if (!is_initialized) return;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9);
    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
    F(0, 6) = 0.5 * dt * dt; F(1, 7) = 0.5 * dt * dt; F(2, 8) = 0.5 * dt * dt;
    F(3, 6) = dt; F(4, 7) = dt; F(5, 8) = dt;

    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double q_a = 50.0; // 加速度过程噪声 (根据实际情况微调)

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(9, 9);
    for(int i = 0; i < 3; i++) {
      Q(i, i) = 0.25 * dt4 * q_a;
      Q(i, i+3) = Q(i+3, i) = 0.5 * dt3 * q_a;
      Q(i, i+6) = Q(i+6, i) = 0.5 * dt2 * q_a;
      Q(i+3, i+3) = dt2 * q_a;
      Q(i+3, i+6) = Q(i+6, i+3) = dt * q_a;
      Q(i+6, i+6) = 1.0 * q_a;
    }

    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd { return F * x; };
    kf.predict(F, Q, f);
  }

  void update(const Eigen::Vector3d & xyz) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 9);
    H.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * 0.1;

    auto h = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd { return x.head(3); };
    auto z_sub = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd { return a - b; };
    kf.update(xyz, H, R, h, z_sub);
    update_count++;
  }
};

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
  std::pair<bool, double> tower_armor_hs[3] = {std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0)};
  double tower_armor_h;
  double tower_armor_hs_datas[3] = {0,0,0}; 
  double last_tower_armor_h[3] = {0,0,0};
  int tower_armor_hs_datas_ptr[3] = {0, 0, 0};

  //长短焦
  bool cam_is_short = true;
  bool last_cam_is_short = true;
  std::chrono::steady_clock::time_point cam_is_switch_time_point; //相机切换时间点；
  
  int update_count_;

private:
  int armor_num_;
  int switch_count_;
  
  bool is_switch_, is_converged_;

  // 单一 EKF 实例
  tools::ExtendedKalmanFilter ekf_; 

  // 【新增】存储当前车辆4个装甲板的独立CA模型滤波器
  std::vector<SingleArmorCA> armor_ca_kfs_;

  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP