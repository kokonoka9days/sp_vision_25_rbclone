#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <optional>
#include <queue>
#include <string>
#include <vector>
#include <cmath>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name = ArmorName::not_armor;
  Color color = Color::extinguish;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;  // debug only
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  bool update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const; 
  Eigen::Vector3d cv_center() const;
  Eigen::Vector3d fused_center() const;
  const Eigen::Vector3d & observed_center() const;
  std::chrono::steady_clock::time_point observation_time() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  inline Eigen::VectorXd getEKFXest() {
    return ekf_.x;
  }

  inline Eigen::VectorXd ca_ekf_x() const { return ca_ekf_.x; }
  inline bool ca_ekf_ready() const { return ca_ekf_init_; }
  inline double get_w_cv() const { return w_cv_; }

  inline std::chrono::steady_clock::time_point getTimePoint() {
    return t_;
  }

  //前哨站
  std::pair<bool, double> tower_armor_hs[3] = {std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0)};
  double tower_armor_h = 0.0;
  double tower_armor_hs_datas[3] = {0,0,0}; 
  double last_tower_armor_h[3] = {0,0,0};
  int tower_armor_hs_datas_ptr[3] = {0, 0, 0};

  //长短焦
  bool cam_is_short = true;
  bool last_cam_is_short = true;
  std::chrono::steady_clock::time_point cam_is_switch_time_point; //相机切换时间点；
  
  int update_count_ = 0;

private:
  int armor_num_ = 4;
  double nominal_radius_ = 0.2;
  int switch_count_ = 0;
  
  bool is_switch_ = false;
  bool is_converged_ = false;

 // 单一整车 EKF (CV模型)
 tools::ExtendedKalmanFilter ekf_; 

  // 中心级 CA EKF (9维: x,vx,ax, y,vy,ay, z,vz,az)
  tools::ExtendedKalmanFilter ca_ekf_;
  bool ca_ekf_init_ = false;
  int ca_update_count_ = 0;
  double w_cv_ = 1.0;
  double cv_error_ema_ = 0.0;
  double ca_error_ema_ = 0.0;
  bool model_error_initialized_ = false;
  double prediction_age_ = 0.0;

  Eigen::Vector3d observed_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d cv_position_at_observation_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ca_position_at_observation_ = Eigen::Vector3d::Zero();
  std::chrono::steady_clock::time_point observation_time_{};

  struct CenterObservation
  {
    std::chrono::steady_clock::time_point time;
    Eigen::Vector3d center;
  };
  std::deque<CenterObservation> center_history_;

  std::chrono::steady_clock::time_point t_;

  void init_ca_filter(const Eigen::Vector3d & center, const Eigen::Vector3d & velocity);
  void clamp_ca_state();
  void record_center_observation(
    std::chrono::steady_clock::time_point time, const Eigen::Vector3d & center);
  Eigen::Vector3d align_with_bearing_prediction(const Eigen::Vector3d & center) const;
  Eigen::Vector3d center_from_armor(const Armor & armor, int id) const;
  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
