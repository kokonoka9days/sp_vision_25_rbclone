#ifndef AUTO_AIM__KF_EXAMPLE__RV_FROM_FYT_HPP
#define AUTO_AIM__KF_EXAMPLE__RV_FROM_FYT_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <utility>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
class RVfromFYT : public tools::ExtendedKalmanFilter
{
public:
  static constexpr Eigen::Index kStateDimension = 11;
  static constexpr double kTowerArmorHeightStep = 0.10;

  RVfromFYT() = default;
  RVfromFYT(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0, int armor_num,
    ArmorName armor_name);

  void predict_model(
    double dt, const Eigen::Vector3d & acceleration, double position_process_noise,
    double yaw_process_noise);

  void prepare_measurement(const Armor & armor, bool cam_is_short, int update_count);
  int select_armor_id(int last_id) const;
  Eigen::VectorXd correct(int armor_id);

  void set_tower_armor_heights(const std::pair<bool, double> (&heights)[3]);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & state, int armor_id) const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

private:
  int armor_num_ = 0;
  ArmorName armor_name_ = ArmorName::not_armor;
  std::array<double, 3> tower_armor_heights_{};

  Eigen::Vector4d z_ = Eigen::Vector4d::Zero();
  Eigen::Matrix4d R_ = Eigen::Matrix4d::Identity();
  bool measurement_ready_ = false;

  bool last_cam_is_short_ = true;
  std::chrono::steady_clock::time_point camera_switch_time_{};

  Eigen::Vector4d h(const Eigen::VectorXd & state, int armor_id) const;
  Eigen::Matrix<double, 4, kStateDimension> h_jacobian(
    const Eigen::VectorXd & state, int armor_id) const;
  double tower_height_multiplier(int armor_id) const;

  static Eigen::VectorXd state_add(
    const Eigen::VectorXd & state, const Eigen::VectorXd & delta);
  static Eigen::VectorXd observation_subtract(
    const Eigen::VectorXd & observation, const Eigen::VectorXd & prediction);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__KF_EXAMPLE__RV_FROM_FYT_HPP
