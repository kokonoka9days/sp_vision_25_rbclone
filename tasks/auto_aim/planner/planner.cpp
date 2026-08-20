#include "planner.hpp"

#include <cmath>
#include <vector>
#include <stdexcept>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
std::vector<tools::YawDelayPoint> read_yaw_delay_curve(
  const YAML::Node & yaml, const char * direction)
{
  const auto curve = yaml["yaw_delay_curve"][direction];
  if (!curve || !curve.IsSequence() || curve.size() == 0) {
    throw std::invalid_argument(std::string("yaw_delay_curve.") + direction + " must be a non-empty sequence");
  }

  std::vector<tools::YawDelayPoint> points;
  points.reserve(curve.size());
  for (const auto & point : curve) {
    if (!point.IsSequence() || point.size() != 2) {
      throw std::invalid_argument(
        std::string("yaw_delay_curve.") + direction + " points must be [speed_rad_s, delay_s]");
    }
    try {
      points.push_back({point[0].as<double>(), point[1].as<double>()});
    } catch (const YAML::Exception & error) {
      throw std::invalid_argument(
        std::string("yaw_delay_curve.") + direction + " contains non-numeric values: " +
        error.what());
    }
  }
  return points;
}
}  // namespace

Planner::Planner(const std::string & config_path) : config_path_(config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  far_pitch_offset_ = tools::read<double>(yaml, "far_pitch_offset") / 57.3;
  far_high_pitch_offset_ = tools::read<double>(yaml, "far_high_pitch_offset") / 57.3;
  target_dist_error_ = tools::read<double>(yaml, "target_dist_error");
  target_h_error_ = tools::read<double>(yaml, "target_h_error");
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  small_armor_tolerance = tools::read<double>(yaml, "small_armor_tolerance");
  big_armor_tolerance = tools::read<double>(yaml, "big_armor_tolerance");
  tower_and_base_armor_tolerance_ = tools::read<double>(yaml, "tower_and_base_armor_tolerance_");
  gimbal_control_delay = tools::read<double>(yaml, "gimbal_control_delay");
  tower_pitch_prediction_time_ = tools::read<double>(yaml, "tower_pitch_prediction_time");
  gimbal_delay_ = tools::read<double>(yaml, "gimbal_delay");
  const auto yaw_delay_curve = yaml["yaw_delay_curve"];
  if (yaw_delay_curve.IsDefined()) {
    if (!yaw_delay_curve.IsMap()) {
      throw std::invalid_argument("yaw_delay_curve must be a map with positive and negative curves");
    }
    try {
      const auto positive = read_yaw_delay_curve(yaml, "positive");
      const auto negative = read_yaw_delay_curve(yaml, "negative");
      const double reverse_penalty = yaml["yaw_reverse_penalty"]
                                       ? yaml["yaw_reverse_penalty"].as<double>()
                                       : 0.0;
      const double deadband = yaml["yaw_direction_deadband"]
                                ? yaml["yaw_direction_deadband"].as<double>()
                                : 0.2;
      const double reverse_window = yaml["yaw_reverse_window"]
                                      ? yaml["yaw_reverse_window"].as<double>()
                                      : 0.05;
      yaw_delay_model_ = tools::YawDelayModel(
        positive, negative, reverse_penalty, deadband, reverse_window);
    } catch (const YAML::Exception & error) {
      throw std::invalid_argument(std::string("invalid yaw delay configuration: ") + error.what());
    }
  }
  shoot_offset_ = tools::read<int>(yaml, "shoot_offset");
  if (!valid_shoot_offset(shoot_offset_)) {
    throw std::invalid_argument("shoot_offset must keep the firing index inside the MPC horizon");
  }

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Planner::Planner(const Planner & other) : Planner(other.config_path_)
{
  debug_xyza = other.debug_xyza;
  aim_target_yaw = other.aim_target_yaw;
  is_far = other.is_far;
  is_high = other.is_high;
  last_selected_idx = other.last_selected_idx;
  last_selected_xyz = other.last_selected_xyz;
  yaw_delay_model_ = other.yaw_delay_model_;
  last_yaw_command_velocity_ = other.last_yaw_command_velocity_;
  outpost_z_stable_start_time_ = other.outpost_z_stable_start_time_;
  outpost_is_make = other.outpost_is_make;
}

Plan Planner::plan(
  std::optional<Target> target, double bullet_speed, double gimbal_yaw,
  ShootStrategy strategy)
{
  if (!target.has_value()) return {};

  const double target_model_delay =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
  if (std::abs(target->ekf_x()[7]) > decision_speed_) {
    tools::logger()->warn(
      "std::abs(target->ekf_x()[7]) > {}", decision_speed_);
  }

  is_far = false;
  is_high = false;
  const auto now = std::chrono::steady_clock::now();
  const double initial_delay =
    strategy == rbSuppressiveFire ? target_model_delay : target_model_delay + gimbal_delay_;
  target->predict(now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(initial_delay)));

  switch (strategy) {
    case Dynamics:
      return plan(*target, bullet_speed);
    case rbSuppressiveFire:
      return rbplan(*target, bullet_speed, gimbal_yaw);
    case rbHero:
      return rbHeroplan(*target, bullet_speed, gimbal_yaw);
    case SB:
      return sbplan(*target, bullet_speed, gimbal_yaw);
    default:
      tools::logger()->error("Unknown shoot strategy: {}", static_cast<int>(strategy));
      return {};
  }
}

Planner & Planner::operator=(const Planner & other)
{
  if (this == &other) return *this;
  Planner replacement(other);
  *this = std::move(replacement);
  return *this;
}

Plan Planner::plan(Target target, double bullet_speed)
{
  if (target.armor_xyza_list().empty()) return {false};
  // std::cout<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<target.getEKFXest()[0]<<std::endl;

  // std::cout<<"x:"<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<"y:"<<target.getEKFXest()[2]<<std::endl;
  // std::cout<<"z:"<<target.getEKFXest()[4]<<std::endl;


  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  min_dist+=target_dist_error_;
  double target_h = xyz.z(); 
  target_h+= target_h_error_;
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, target_h);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_.get(), x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_.get());

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_.get(), x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_.get());

  Plan plan{};
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);

  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  // auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}


Plan Planner::sbplan(Target target, double bullet_speed, double gimbal_yaw)
{
  if (target.armor_xyza_list().empty()) return {false};
  // std::cout<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<target.getEKFXest()[0]<<std::endl;

  // std::cout<<"x:"<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<"y:"<<target.getEKFXest()[2]<<std::endl;
  // std::cout<<"z:"<<target.getEKFXest()[4]<<std::endl;


  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  min_dist+=target_dist_error_;
  double target_h = xyz.z(); 
  target_h+= target_h_error_;
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, target_h);
  is_far = target_h > 1.0;
  
  target.predict(bullet_traj.fly_time);

  // tools::logger()->info("h:{}, xy_d:{}, xyz_d:{}, fly_time:{}, ", target_h, min_dist, xyz.norm(), bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  Eigen::Vector2d yaw_pitch;
  try {
    yaw_pitch = rbaim(target, bullet_speed);
    yaw0 = yaw_pitch(0);
    traj = rbget_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_.get(), x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_.get());

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_.get(), x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_.get());

  Plan plan{};
  plan.control = true;
  //mubiaojaiiodu
  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  

  // 前哨站迭代限制
  if (target.name == ArmorName::outpost) {
      double vz = target.ekf_x()(5); // 获取当前前哨站中心Z轴坐标
      // double delta_z = std::abs(current_z - outpost_z_baseline_);
      auto now = std::chrono::steady_clock::now();

      // 如果Z轴变化幅度大于指定阈值（例如0.05米），重置基准和计时器，并禁止开火
      if (vz > 0.09) { 
          outpost_z_stable_start_time_ = now;
          // suggest_fire = false; 
          outpost_is_make = false;
      } else {
        // 如果变化幅度在阈值内，判断持续时间是否达到 0.7 秒
        double stable_duration = tools::delta_time(now ,outpost_z_stable_start_time_ );
        if (
          // stable_duration < 1 || 
          target.update_count_ < 500) {
            // suggest_fire = false; // 持续时间不足 0.7s，不开火
            outpost_is_make = false;
        }
        else{
          outpost_is_make = true;
        }
      }

      if(!outpost_is_make){
        Eigen::Vector2d yaw_pitch_nan = heroaim(target, 100000, gimbal_yaw);
        plan.yaw = yaw_pitch_nan(0);
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;

        plan.pitch = yaw_pitch_nan(1);
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
        plan.fire = 0;
        return  plan;
      }
  }

  // auto shoot_offset_ = 2;
  if(abs(tools::limit_rad((gimbal_yaw )/57.3 - yaw_offset_ - plan.target_yaw)) * 57.3 < 3){
    plan.fire =
      std::hypot(
        traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
        traj(2, HALF_HORIZON + shoot_offset_) -
          pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  }else plan.fire = 0;

  // target.predict(-gimbal_control_delay);
  // plan.fire = rbShoot(target, (gimbal_yaw )/57.3 - yaw_offset_);
  // tools::logger()->warn("fire:{}", plan.fire);
  // plan.target_yaw = (aim_target_yaw + yaw_offset_ )* 57.3;




  return plan;
}




bool Planner::rbShoot(Target target, double gimbal_yaw, bool tower_fixed_pitch){
  if (target.armor_xyza_list().empty()) return false;
  bool suggest_fire = 1;
    // auto x_est = target.getEKFXest();
    // double est_x =  x_est(0);
    // double est_y = x_est(2);
    // double est_yaw = x_est(6);


  Eigen::Vector4d target_armor_xyza;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      target_armor_xyza = xyza;
    }
  }

  double target_yaw = target_armor_xyza(3) ;

  aim_target_yaw = atan2(target_armor_xyza(1), target_armor_xyza(0));//+ 0.3/57.3;
  // feedback_yaw = gimbal_yaw;

  double shoot_range = target.armor_type == ArmorType::big ? big_armor_tolerance : small_armor_tolerance;

  if(target.name == ArmorName::base || target.name == ArmorName::outpost) shoot_range = tower_and_base_armor_tolerance_;

    // 打击范围计算

  // 左边缘（沿切线正方向偏移半宽）
  double left_x = target_armor_xyza(0) + 0.5 * shoot_range * (-sin(target_yaw));
  double left_y = target_armor_xyza(1) + 0.5 * shoot_range * cos(target_yaw);

  // 右边缘（沿切线负方向偏移半宽）
  double right_x = target_armor_xyza(0) - 0.5 * shoot_range * (-sin(target_yaw));
  double right_y = target_armor_xyza(1) - 0.5 * shoot_range * cos(target_yaw);


  // 目标中心方向
  double center_angle = atan2(target_armor_xyza(1), target_armor_xyza(0));
  // 当前云台偏差（相对于中心）
  double delta = tools::limit_rad(center_angle - gimbal_yaw);


  double left_angle = atan2(left_y, left_x);
  double right_angle = atan2(right_y, right_x);
  // 计算左右边缘相对于中心的角度偏移
  double d_left = tools::limit_rad(left_angle - center_angle);
  double d_right = tools::limit_rad(right_angle - center_angle);

  // 取较小的和较大的偏移（因为左右边缘距离中心不会超过 90°，所以 d_left 和 d_right 符号相反且绝对值 < π/2）
  double d_min = std::min(d_left, d_right);
  double d_max = std::max(d_left, d_right);



  // double ax = target_armor_xyza(0) - 0.5f * shoot_range * sin(target_yaw);
  // double ay = target_armor_xyza(1) + 0.5f * shoot_range * cos(target_yaw);
  // double bx = target_armor_xyza(0) + 0.5f * shoot_range * sin(target_yaw);
  // double by = target_armor_xyza(1) - 0.5f * shoot_range * cos(target_yaw);
  // double angle_a = atan2(ay, ax);
  // double angle_b = atan2(by, bx);
  // double angle_c = atan2(target_armor_xyza(1), target_armor_xyza(0));
  // // double allow_fire_ang_max = angle_c - angle_b;
  // // double allow_fire_ang_min = angle_c - angle_a;
  // double allow_fire_ang_max = std::max(angle_a, angle_b) - angle_c;
  // double allow_fire_ang_min = std::min(angle_a, angle_b) - angle_c;
  // allow_fire_ang_max = tools::limit_rad(allow_fire_ang_max);
  // allow_fire_ang_min = tools::limit_rad(allow_fire_ang_min);
  

  // pitch
  bool suggest_pitch = true;
  if(tower_fixed_pitch && abs(target.ekf_x()(4) - target_armor_xyza(2)) > 0.001){
    suggest_pitch = false;
  }

  // // yaw_ang_ref
  // double control_delta_angle =
  //     tools::limit_rad(atan2(target_armor_xyza(1), target_armor_xyza(0)) - gimbal_yaw );
  // suggest_fire = (control_delta_angle < allow_fire_ang_max &&
  //                 control_delta_angle > allow_fire_ang_min && suggest_pitch) ;

  // 判断 delta 是否在 [d_min, d_max] 范围内
  suggest_fire = (delta >= d_min && delta <= d_max) && suggest_pitch;



  if(!outpost_is_make && target.name == ArmorName::outpost) suggest_fire = 0;
  if(!suggest_fire){
    // tools::logger()->info("not fire! control_delta_angle: {},  allow_fire_ang_max: {}, allow_fire_ang_min: {}",
    //   control_delta_angle, allow_fire_ang_max, allow_fire_ang_min
    // );
    
  }
    
    return suggest_fire;
}

Plan Planner::rbplan(Target target, double bullet_speed, double gimbal_yaw)
{
  if (target.armor_xyza_list().empty()) return {false};
  // std::cout<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<target.getEKFXest()[0]<<std::endl;

  // std::cout<<"x:"<<target.getEKFXest()[0]<<std::endl;
  // std::cout<<"y:"<<target.getEKFXest()[2]<<std::endl;
  // std::cout<<"z:"<<target.getEKFXest()[4]<<std::endl;


  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  min_dist+=target_dist_error_;
  double target_h = xyz.z(); 
  target_h+= target_h_error_;
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, target_h);
  
  is_far = min_dist > 5.0;
  is_high = target_h > 1.3;

  tools::logger()->info("h:{}, xy_d:{}, xyz_d:{}, fly_time:{}, ", target_h, min_dist, xyz.norm(), bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  const auto now = std::chrono::steady_clock::now();
  const double pitch_delay = gimbal_delay_;
  double yaw_delay = gimbal_delay_;
  double yaw_reference_velocity = 0;
  Target pitch_target = target;
  pitch_target.predict(pitch_delay + bullet_traj.fly_time);
  try {
    for (int iteration = 0; iteration < 3; ++iteration) {
      Target yaw_target = target;
      yaw_target.predict(yaw_delay + bullet_traj.fly_time);
      yaw0 = aim(yaw_target, bullet_speed)(0);
      traj = rbget_trajectory_split(yaw_target, pitch_target, yaw0, bullet_speed);
      if (!traj.allFinite()) throw std::runtime_error("non-finite yaw/pitch reference trajectory");

      if (!yaw_delay_model_.enabled()) break;
      // Use the forward difference of the generated yaw reference. This is
      // the command velocity seen by the gimbal, rather than the target EKF's
      // angular velocity.
      yaw_reference_velocity = tools::limit_rad(
        traj(0, HALF_HORIZON + 1) - traj(0, HALF_HORIZON)) / DT;
      auto candidate_model = yaw_delay_model_;
      const double updated_delay = candidate_model.query(
        yaw_reference_velocity, last_yaw_command_velocity_, now);
      if (!std::isfinite(updated_delay) || updated_delay < 0 || updated_delay > 0.2) {
        throw std::runtime_error("invalid yaw delay query result");
      }
      if (std::abs(updated_delay - yaw_delay) < 0.001 || iteration == 2) {
        // On the final allowed pass keep the delay that generated `traj`, so
        // the reported delay and reference trajectory remain consistent.
        if (std::abs(updated_delay - yaw_delay) < 0.001) yaw_delay = updated_delay;
        break;
      }
      yaw_delay = updated_delay;
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}: {}", bullet_speed, e.what());
    return {false};
  }

  if (!traj.allFinite() || !std::isfinite(yaw_delay)) {
    tools::logger()->warn("Invalid rb trajectory or yaw delay");
    return {false};
  }
  target = pitch_target;

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  if (
    !yaw_solver_ || !yaw_solver_->work || tiny_set_x0(yaw_solver_.get(), x0) != 0) {
    tools::logger()->warn("Yaw TinyMPC setup failed for rb target");
    return {false};
  }
  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  if (
    tiny_solve(yaw_solver_.get()) != 0 ||
    !yaw_solver_->work->x.allFinite() || !yaw_solver_->work->u.allFinite()) {
    tools::logger()->warn("Yaw TinyMPC failed for rb target");
    return {false};
  }

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  if (
    !pitch_solver_ || !pitch_solver_->work || tiny_set_x0(pitch_solver_.get(), x0) != 0) {
    tools::logger()->warn("Pitch TinyMPC setup failed for rb target");
    return {false};
  }
  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  if (
    tiny_solve(pitch_solver_.get()) != 0 ||
    !pitch_solver_->work->x.allFinite() || !pitch_solver_->work->u.allFinite()) {
    tools::logger()->warn("Pitch TinyMPC failed for rb target");
    return {false};
  }

  Plan plan{};
  plan.control = true;
  //mubiaojaiiodu
  // plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);
  if (
    !std::isfinite(plan.yaw) || !std::isfinite(plan.yaw_vel) || !std::isfinite(plan.yaw_acc) ||
    !std::isfinite(plan.pitch) || !std::isfinite(plan.pitch_vel) ||
    !std::isfinite(plan.pitch_acc)) {
    tools::logger()->warn("rb TinyMPC returned non-finite control");
    return {false};
  }
  plan.yaw_delay = static_cast<float>(yaw_delay);
  plan.yaw_delay_direction = yaw_delay_model_.enabled()
                               ? yaw_delay_model_.direction(yaw_reference_velocity)
                               : 0;

  // 前哨站迭代限制
  if (target.name == ArmorName::outpost) {
      double vz = target.ekf_x()(5); // 获取当前前哨站中心Z轴坐标
      // double delta_z = std::abs(current_z - outpost_z_baseline_);
      auto now = std::chrono::steady_clock::now();

      // 如果Z轴变化幅度大于指定阈值（例如0.05米），重置基准和计时器，并禁止开火
      // if (vz > 0.09) { 
      //     outpost_z_stable_start_time_ = now;
      //     // suggest_fire = false; 
      //     outpost_is_make = false;
      // } else {
        // 如果变化幅度在阈值内，判断持续时间是否达到 0.7 秒
        double stable_duration = tools::delta_time(now ,outpost_z_stable_start_time_ );
        if (
          // stable_duration < 1 || 
          target.update_count_ < 500) {
            // suggest_fire = false; // 持续时间不足 0.7s，不开火
            outpost_is_make = false;
        }
        else{
          outpost_is_make = true;
        }
      // }

      if(!outpost_is_make){
        Eigen::Vector2d yaw_pitch_nan = heroaim(target, 100000, gimbal_yaw);
        plan.yaw = yaw_pitch_nan(0);
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;

        plan.pitch = yaw_pitch_nan(1);
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
      }
  }

  if (
    !std::isfinite(plan.yaw) || !std::isfinite(plan.yaw_vel) || !std::isfinite(plan.yaw_acc) ||
    !std::isfinite(plan.pitch) || !std::isfinite(plan.pitch_vel) ||
    !std::isfinite(plan.pitch_acc)) {
    tools::logger()->warn("rb control override returned non-finite values");
    return {false};
  }
  if (yaw_delay_model_.enabled()) {
    // Commit direction/reversal state using the command that was actually
    // emitted by MPC (including any outpost safety override).
    try {
      yaw_delay_model_.query(plan.yaw_vel, last_yaw_command_velocity_, now);
    } catch (const std::exception & e) {
      tools::logger()->warn("Yaw delay state update failed: {}", e.what());
      yaw_delay_model_.reset();
    }
    plan.yaw_reversing = yaw_delay_model_.reversal_active(now);
  }
  last_yaw_command_velocity_ = plan.yaw_vel;

  // 开火判断依据
  auto is_fire = [this](const double plan_yaw, const Target& target_, bool tower_fixed_pitch){
    bool suggest_fire = 1;

    auto xyzad = target_.get_recent_armor_xyzad();
    Eigen::Vector4d target_armor_xyza = xyzad.head<4>();
    double target_yaw = target_armor_xyza(3) ;
    aim_target_yaw = atan2(target_armor_xyza(1), target_armor_xyza(0));//+ 0.3/57.3;
    double shoot_range = target_.armor_type == ArmorType::big ? big_armor_tolerance : small_armor_tolerance;

    if(target_.name == ArmorName::base || target_.name == ArmorName::outpost) 
      shoot_range = tower_and_base_armor_tolerance_;

      
    // 打击范围计算
    // 左边缘（沿切线正方向偏移半宽）
    double left_x = target_armor_xyza(0) + 0.5 * shoot_range * (-sin(target_yaw));
    double left_y = target_armor_xyza(1) + 0.5 * shoot_range * cos(target_yaw);

    // 右边缘（沿切线负方向偏移半宽）
    double right_x = target_armor_xyza(0) - 0.5 * shoot_range * (-sin(target_yaw));
    double right_y = target_armor_xyza(1) - 0.5 * shoot_range * cos(target_yaw);


    // 目标中心方向
    double center_angle = atan2(target_armor_xyza(1), target_armor_xyza(0));
    // 当前云台偏差（相对于中心）
    double delta = tools::limit_rad(plan_yaw - center_angle);


    double left_angle = atan2(left_y, left_x);
    double right_angle = atan2(right_y, right_x);
    // 计算左右边缘相对于中心的角度偏移
    double d_left = tools::limit_rad(left_angle - center_angle);
    double d_right = tools::limit_rad(right_angle - center_angle);

    // 取较小的和较大的偏移（因为左右边缘距离中心不会超过 90°，所以 d_left 和 d_right 符号相反且绝对值 < π/2）
    double d_min = std::min(d_left, d_right);
    double d_max = std::max(d_left, d_right);


    // pitch
    bool suggest_pitch = true;
    if(tower_fixed_pitch && abs(target_.ekf_x()(4) - target_armor_xyza(2)) > 0.001){
      suggest_pitch = false;
    }
    // 判断 delta 是否在 [d_min, d_max] 范围内
    suggest_fire = (delta >= d_min && delta <= d_max) && suggest_pitch;



    if(!outpost_is_make && target_.name == ArmorName::outpost) suggest_fire = 0;
    if(!suggest_fire){
      // tools::logger()->info("not fire! control_delta_angle: {},  allow_fire_ang_max: {}, allow_fire_ang_min: {}",
      //   control_delta_angle, allow_fire_ang_max, allow_fire_ang_min
      // );
    }
      
    return suggest_fire;
  };
  plan.fire = is_fire(plan.yaw - yaw_offset_, target, false);
  // tools::logger()->warn("fire:{}", plan.fire);
  plan.target_yaw = (aim_target_yaw + yaw_offset_ )* 57.3;

  return plan;
}


Plan Planner::rbHeroplan(Target target, double bullet_speed, double gimbal_yaw){
  if (target.armor_xyza_list().empty()) return {false};
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  min_dist+=target_dist_error_;
  double target_h = xyz.z(); 
  target_h+= target_h_error_;

    // tools::logger()->info("h:{}, xy_d:{}, xyz_d:{}, fly_time:{}, ", xyz.z(), min_dist, xyz.norm(), bullet_traj.fly_time);

  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, target_h);
  is_far = min_dist > 5.0;
//  tools::logger()->info("h:{}, xy_d:{}, xyz_d:{}, fly_time:{}, is_far{} ", target_h, min_dist, xyz.norm(), bullet_traj.fly_time, is_far);
  
  target.predict(bullet_traj.fly_time );

  // 2. Get trajectory
  Eigen::Vector2d yaw_pitch;
  try {
    yaw_pitch = heroaim(target, bullet_speed, gimbal_yaw);
    if (!yaw_pitch.allFinite()) throw std::runtime_error("non-finite hero aim result");
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }



  Plan plan{};
  plan.control = true;
  // plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  
  plan.target_pitch = yaw_pitch(1);

  plan.yaw = yaw_pitch(0); // tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = 0; //yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = 0; //yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = yaw_pitch(1); //itch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = 0; //pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = 0; //pitch_solver_->work->u(0, HALF_HORIZON);

  
  // plan.fire =
  //   std::hypot(
  //     traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
  //     traj(2, HALF_HORIZON + shoot_offset_) -
  //       pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  target.predict(-gimbal_control_delay);
  plan.fire = rbShoot(target, (gimbal_yaw )/57.3 - yaw_offset_
                                                    ,true
                                                  );
  // tools::logger()->warn("fire:{}", plan.fire);
  plan.target_yaw = (aim_target_yaw + yaw_offset_ )* 57.3;



  // 前哨站迭代限制
  if (target.name == ArmorName::outpost) {
      double vz = target.ekf_x()(5); // 获取当前前哨站中心Z轴坐标
      // double delta_z = std::abs(current_z - outpost_z_baseline_);
      auto now = std::chrono::steady_clock::now();

      // 如果Z轴变化幅度大于指定阈值（例如0.05米），重置基准和计时器，并禁止开火
      if (vz > 0.01) { 
          outpost_z_stable_start_time_ = now;
          // suggest_fire = false; 
          outpost_is_make = false;
      } else {
        if (
          target.update_count_ < 500) {
            // suggest_fire = false; // 持续时间不足 0.7s，不开火
            outpost_is_make = false;
        }
        else{
          outpost_is_make = true;
        }
      }

      if(!outpost_is_make){
        Eigen::Vector2d yaw_pitch_nan = heroaim(target, 100000, gimbal_yaw);
        plan.yaw = yaw_pitch_nan(0);
        plan.yaw_vel = 0;
        plan.yaw_acc = 0;

        plan.pitch = yaw_pitch_nan(1) + 10/57.3;
        plan.pitch_vel = 0;
        plan.pitch_acc = 0;
      }
  }

  return plan;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  if (target.armor_xyza_list().empty()) throw std::runtime_error("Target has no armor pose");
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), bullet_traj.pitch + pitch_offset_};
}

Eigen::Matrix<double, 2, 1> Planner::rbaim(const Target & target, double bullet_speed)
{
  if (target.armor_xyza_list().empty()) throw std::runtime_error("Target has no armor pose");

  Eigen::Matrix<double, 5, 1> xyzad = target.get_recent_armor_xyzad();
  Eigen::Vector3d xyz = xyzad.head<3>();
  double yaw = xyzad(3);
  auto min_dist = xyz.head<2>().norm();

  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  double now_pitch_offset = 0;
  if(is_far & is_high) {now_pitch_offset = far_high_pitch_offset_;
    // tools::logger()->info("far_high_pitch_offset_");
  }
  else if(is_far) {
    now_pitch_offset = far_pitch_offset_;
    // tools::logger()->info("far_pitch_offset_");
  }
  else {now_pitch_offset = pitch_offset_; 
    // tools::logger()->info("pitch_offset_");
  }

  return {tools::limit_rad(azim + yaw_offset_), bullet_traj.pitch + now_pitch_offset};


}

Eigen::Matrix<double, 2, 1> Planner::heroaim(const Target & target, double bullet_speed, double gimbal_yaw)
{
  auto armors = target.armor_xyza_list();
  if (armors.empty()) throw std::runtime_error("Target has no armor pose");

  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  Eigen::VectorXd ekf_x = target.ekf_x();
  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<std::pair<int ,double>> armorId_delta_list;
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();

  auto armor_num = armor_xyza_list.size();
  // // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  // if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    // auto dist = armor_xyza_list[i].head<2>().norm();
    armorId_delta_list.emplace_back(std::make_pair(i, delta_angle));
  }
  
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }

  double abs_vyaw = abs(ekf_x(7));
  if(target.last_id >= 0 && static_cast<std::size_t>(target.last_id) < armor_xyza_list.size() &&
    abs_vyaw < 90./57.3 &&
    armorId_delta_list[target.last_id].second < 60./57.3){// 判断当前看到的装甲板在预测时间之后是否还在视野内
    min_dist = armor_xyza_list[target.last_id].head<2>().norm();
    xyz = armor_xyza_list[target.last_id].head<3>();
    yaw = armor_xyza_list[target.last_id](3);
  }



  auto r = target.ekf_x()(8);
  auto v_yaw = target.ekf_x()(7);

    // 旋转中心的坐标
  auto center_x = target.ekf_x()(0);
  auto center_y = target.ekf_x()(2);

  auto direction_yaw = atan2(center_y, center_x);
  auto aim_point_x = center_x - r*std::cos(direction_yaw);
  auto aim_point_y = center_y - r*std::sin(direction_yaw);
  auto aim_point_z = xyz.z();

  if(abs(v_yaw) < 1){
    aim_point_x = xyz.x();
    aim_point_y = xyz.y();
  }
  auto min_dist1 = min_dist;
  if(target.name == ArmorName::outpost){
    if (armor_xyza_list.size() < 3) throw std::runtime_error("Outpost target requires three armor poses");
    Target target_pitch = target;
    min_dist1 = 1e10;
    Eigen::Vector3d xyz1;
    double yaw1;
    double max_h_armor = 10e-6, min_h_armor = 10e+6;
    size_t max_armor_id, min_armor_id;
    target_pitch.predict(tower_pitch_prediction_time_);
    // for (auto & xyza : target.armor_xyza_list()) {
    //   auto dist = xyza.head<2>().norm();
    //   if (dist < min_dist1) {
    //     min_dist1 = dist;
    //     xyz1 = xyza.head<3>();
    //     yaw1 = xyza[3];
    //   }
    //   if(max_h_armor < xyza(2)) {
    //     max_h_armor = xyza(2); 
    //   }
    // }
    for(int i = 0; i < 3; i++){
      auto xyza = armor_xyza_list[i];
      auto dist = xyza.head<2>().norm();
      if (dist < min_dist1) {
        min_dist1 = dist;
        xyz1 = xyza.head<3>();
        yaw1 = xyza[3];
      }
      if(max_h_armor < xyza(2)) {
        max_h_armor = xyza(2); 
        max_armor_id = i;
      }
      if(min_h_armor > xyza(2)){
        min_h_armor = xyza(2); 
        min_armor_id = i;
      }
    }
    size_t middle_armor_id = 0;
    for(int i = 0; i < 3; i++){
      if(min_armor_id != i && max_armor_id != i ) middle_armor_id = i;
    }
    // if(abs(max_h_armor - target.ekf_x()(4)) < 0.05) aim_point_z = 
    // else aim_point_z = target.ekf_x()(4);
    aim_point_z = armor_xyza_list[middle_armor_id](2);
  }
  
  //补偿距离和补偿高度
  double comp_dist = 0;
  double comp_h = 0;
  min_dist = sqrt(aim_point_x*aim_point_x + aim_point_y*aim_point_y);

  debug_xyza = Eigen::Vector4d(aim_point_x, aim_point_y, aim_point_z, yaw);

  auto azim = std::atan2(aim_point_y, aim_point_x);
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist1 - comp_dist, aim_point_z - comp_h);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  auto now_pitch_offset = is_far ? far_pitch_offset_ : pitch_offset_;

  return {tools::limit_rad(azim + yaw_offset_), bullet_traj.pitch + now_pitch_offset};
}


}  // namespace auto_aim
