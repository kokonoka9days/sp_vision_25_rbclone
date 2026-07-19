#include "drone_planner.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>  // 需要使用 std::atan2 和 std::hypot
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{
namespace
{

Eigen::Vector3d read_vector3(const YAML::Node & yaml, const std::string & key)
{
  if (!yaml[key]) throw std::runtime_error("Missing laser calibration key: " + key);
  const auto values = yaml[key].as<std::vector<double>>();
  if (values.size() != 3)
    throw std::runtime_error("Laser calibration key must have 3 values: " + key);
  const Eigen::Vector3d result(values[0], values[1], values[2]);
  if (!result.array().isFinite().all()) {
    throw std::runtime_error("Laser calibration key contains non-finite values: " + key);
  }
  return result;
}

Eigen::Matrix3d read_rotation3(const YAML::Node & yaml, const std::string & key)
{
  if (!yaml[key]) throw std::runtime_error("Missing transform key: " + key);
  const auto values = yaml[key].as<std::vector<double>>();
  if (values.size() != 9) throw std::runtime_error("Transform key must have 9 values: " + key);
  return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(values.data());
}

YAML::Node load_runtime_laser_ray(
  const YAML::Node & runtime_yaml, const std::string & config_path, bool & from_external_file,
  std::string & source)
{
  from_external_file = runtime_yaml["laser_ray_config_path"] &&
                       !runtime_yaml["laser_ray_config_path"].as<std::string>().empty();
  if (!from_external_file) {
    source = config_path;
    return runtime_yaml;
  }

  namespace fs = std::filesystem;
  fs::path ray_path = runtime_yaml["laser_ray_config_path"].as<std::string>();
  if (ray_path.is_relative()) {
    const fs::path relative_to_config = fs::absolute(config_path).parent_path() / ray_path;
    ray_path = fs::exists(relative_to_config) ? relative_to_config : fs::absolute(ray_path);
  }
  source = ray_path.lexically_normal().string();
  const YAML::Node ray_yaml = YAML::LoadFile(source);
  if (!ray_yaml["quality"] || ray_yaml["quality"].as<std::string>() != "passed") {
    throw std::runtime_error("Laser ray file did not pass quality gate: " + source);
  }
  return ray_yaml;
}

}  // namespace

Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  auto xyz_offset_vec = tools::read<std::vector<double>>(yaml, "xyz_offset");
  if (xyz_offset_vec.size() == 3) {
    xyz_offset_ = Eigen::Vector3d(xyz_offset_vec[0], xyz_offset_vec[1], xyz_offset_vec[2]);
  } else {
    xyz_offset_ = Eigen::Vector3d::Zero();
  }
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  gimbal_control_delay = tools::read<double>(yaml, "gimbal_control_delay");

  const bool has_inline_ray =
    yaml["laser_line_origin_in_camera_m"] && yaml["laser_line_direction_in_camera"];
  const bool has_ray_path =
    yaml["laser_ray_config_path"] && !yaml["laser_ray_config_path"].as<std::string>().empty();
  laser_ray_enabled_ = yaml["laser_ray_enabled"] ? yaml["laser_ray_enabled"].as<bool>()
                                                 : has_inline_ray || has_ray_path;
  if (laser_ray_enabled_) {
    bool from_external_file = false;
    std::string source;
    const YAML::Node ray_yaml =
      load_runtime_laser_ray(yaml, config_path, from_external_file, source);
    if (!from_external_file && !has_inline_ray) {
      throw std::runtime_error(
        "laser_ray_enabled is true but no inline ray or laser_ray_config_path was provided");
    }

    const Eigen::Vector3d origin_camera = read_vector3(ray_yaml, "laser_line_origin_in_camera_m");
    const Eigen::Vector3d direction_camera =
      read_vector3(ray_yaml, "laser_line_direction_in_camera");
    const Eigen::Matrix3d R_camera2gimbal = read_rotation3(yaml, "R_camera2gimbal");
    const Eigen::Vector3d t_camera2gimbal = read_vector3(yaml, "t_camera2gimbal");

    laser_ray_.direction_in_gimbal = R_camera2gimbal * direction_camera;
    const double direction_norm = laser_ray_.direction_in_gimbal.norm();
    if (!std::isfinite(direction_norm) || direction_norm < 1e-9) {
      throw std::runtime_error("Invalid laser line direction after camera-to-gimbal transform");
    }
    laser_ray_.direction_in_gimbal /= direction_norm;
    const Eigen::Vector3d transformed_origin = R_camera2gimbal * origin_camera + t_camera2gimbal;
    laser_ray_.origin_in_gimbal_m =
      transformed_origin -
      laser_ray_.direction_in_gimbal * laser_ray_.direction_in_gimbal.dot(transformed_origin);
    if (laser_ray_.direction_in_gimbal.x() <= 0.0) {
      throw std::runtime_error("Calibrated laser direction does not point toward gimbal +X");
    }
    tools::logger()->info(
      "[Planner] Laser ray enabled from {}: origin_g=[{:.6f}, {:.6f}, {:.6f}]m, "
      "direction_g=[{:.6f}, {:.6f}, {:.6f}]",
      source, laser_ray_.origin_in_gimbal_m.x(), laser_ray_.origin_in_gimbal_m.y(),
      laser_ray_.origin_in_gimbal_m.z(), laser_ray_.direction_in_gimbal.x(),
      laser_ray_.direction_in_gimbal.y(), laser_ray_.direction_in_gimbal.z());
  } else {
    tools::logger()->warn(
      "[Planner] Laser ray calibration disabled; using the legacy +X ray through gimbal origin");
  }

  // 初始化 MPC 求解器矩阵[cite: 1]
  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 1. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target");
    return {false};
  }

  // 2. Solve yaw[cite: 1]
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 3. Solve pitch[cite: 1]
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;

  return plan;
}

PlanDiagnostics Planner::plan_diagnostics(std::optional<Target> target, double bullet_speed)
{
  PlanDiagnostics diagnostics;
  diagnostics.target_present = target.has_value();
  if (!target.has_value()) return diagnostics;

  diagnostics.input_xyz = target->get_xyz();
  diagnostics.input_velocity = target->get_v();
  try {
    diagnostics.input_yaw_pitch = aim(*target, bullet_speed);
    diagnostics.aim_valid = true;
  } catch (const std::exception &) {
    return diagnostics;
  }
  diagnostics.timestamps_valid =
    target->state_timestamp() != std::chrono::steady_clock::time_point{} &&
    target->last_observation_timestamp() != std::chrono::steady_clock::time_point{};
  if (!diagnostics.timestamps_valid) return diagnostics;

  const auto now = std::chrono::steady_clock::now();
  diagnostics.observation_age_s =
    std::chrono::duration<double>(now - target->last_observation_timestamp()).count();
  diagnostics.target_age_valid =
    diagnostics.observation_age_s >= 0.0 && diagnostics.observation_age_s <= max_target_age_;
  if (!diagnostics.target_age_valid) return diagnostics;

  diagnostics.state_age_s =
    std::max(0.0, std::chrono::duration<double>(now - target->state_timestamp()).count());
  diagnostics.prediction_horizon_s = diagnostics.state_age_s + gimbal_control_delay;
  diagnostics.prediction_target_timestamp =
    target->state_timestamp() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                  std::chrono::duration<double>(diagnostics.prediction_horizon_s));

  target->predict(diagnostics.prediction_horizon_s);
  diagnostics.predicted_xyz = target->get_xyz();
  try {
    diagnostics.predicted_yaw_pitch = aim(*target, bullet_speed);
  } catch (const std::exception &) {
    diagnostics.aim_valid = false;
    return diagnostics;
  }
  diagnostics.plan = plan(*target, bullet_speed);
  diagnostics.plan_valid = diagnostics.plan.control;
  return diagnostics;
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double /*bullet_speed*/)
{
  Eigen::Vector3d xyz = target.get_xyz() + xyz_offset_;

  // 无人机无额外 Yaw 朝向，调试用赋值 0.0[cite: 1]
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), 0.0);

  double azim = std::atan2(xyz.y(), xyz.x());
  double pitch = std::atan2(xyz.z(), xyz.head<2>().norm());
  if (laser_ray_enabled_) {
    const auto solution = solve_laser_ray_aim(xyz, laser_ray_);
    if (!solution) throw std::runtime_error("Target is unreachable by the calibrated laser ray");
    azim = solution->yaw;
    pitch = solution->pitch;
  }

  return {tools::limit_rad(azim + yaw_offset_), pitch + pitch_offset_};
}

Trajectory Planner::get_trajectory(Target target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_drone
