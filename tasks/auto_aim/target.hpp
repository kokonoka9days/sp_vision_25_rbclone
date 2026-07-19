#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <chrono>
#include <optional>
#include <utility>
#include <vector>

#include "armor.hpp"
#include "motion_model.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name = ArmorName::not_armor;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;
  Eigen::Vector3d xyz_in_world = Eigen::Vector3d::Zero();
  bool cam_is_short = true;
  bool last_cam_is_short = true;
  int update_count_ = 0;

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point time,
    const EstimatorConfig & config = {});

  // Compatibility constructor retained for external tests and tools.
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point time, double radius,
    int armor_num, Eigen::VectorXd initial_covariance_diagonal);
  Target(double x, double vyaw, double radius, double height);

  void predict(std::chrono::steady_clock::time_point time);
  void predict(double dt);

  int update(
    const std::vector<std::pair<int, Armor>> & matched_armors, int primary_id,
    const CameraContext & camera, std::chrono::steady_clock::time_point time);

  Eigen::VectorXd ekf_x() const;
  Eigen::VectorXd getEKFXest() const { return ekf_x(); }
  std::chrono::steady_clock::time_point getTimePoint() const { return time_; }
  const motion_model::Covariance & covariance() const;
  tools::EstimatorDiagnostics estimator_diagnostics() const;

  Eigen::Vector3d center() const;
  Eigen::Vector3d velocity() const;
  Eigen::Matrix3d rotation() const;
  double yaw() const;
  double yaw_rate() const;
  double radius(int id) const;
  double armor_height(int id) const;
  int armor_count() const;
  std::vector<Eigen::Isometry3d> armor_pose_list() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool matching_initialized() const;
  bool diverged() const;
  bool convergened();
  bool checkinit() const { return initialized_; }

private:
  EstimatorConfig config_;
  std::optional<motion_model::RobotESEKF> estimator_;
  std::chrono::steady_clock::time_point time_{};
  motion_model::DirectionVoter voter_;
  std::vector<bool> observed_ids_;
  bool initialized_ = false;
  bool converged_ = false;

  void initialize(
    const Armor & armor, std::chrono::steady_clock::time_point time, double initial_radius,
    const std::optional<Eigen::VectorXd> & covariance_diagonal = std::nullopt);
  motion_model::Covariance process_noise(double dt) const;
  void clamp_nominal_state();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
