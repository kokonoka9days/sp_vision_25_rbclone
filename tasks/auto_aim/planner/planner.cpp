#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  small_armor_tolerance = tools::read<double>(yaml, "small_armor_tolerance");
  big_armor_tolerance = tools::read<double>(yaml, "big_armor_tolerance");
  gimbal_control_delay = tools::read<double>(yaml, "gimbal_control_delay");
  tower_pitch_prediction_time_ = tools::read<double>(yaml, "tower_pitch_prediction_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
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
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

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
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
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



bool Planner::rbShoot(Target target, double gimbal_yaw){
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

  double target_yaw = target_armor_xyza(3);

  aim_target_yaw = atan2(target_armor_xyza(1), target_armor_xyza(0));
  // feedback_yaw = gimbal_yaw;

  double shoot_range = target.armor_type == ArmorType::big ? big_armor_tolerance : small_armor_tolerance;

    // 打击范围计算
  double ax = target_armor_xyza(0) - 0.5f * shoot_range * sin(target_yaw);
  double ay = target_armor_xyza(1) + 0.5f * shoot_range * cos(target_yaw);
  double bx = target_armor_xyza(0) + 0.5f * shoot_range * sin(target_yaw);
  double by = target_armor_xyza(1) - 0.5f * shoot_range * cos(target_yaw);
  double angle_a = atan2(ay, ax);
  double angle_b = atan2(by, bx);
  double angle_c = atan2(target_armor_xyza(1), target_armor_xyza(0));
  // double allow_fire_ang_max = angle_c - angle_b;
  // double allow_fire_ang_min = angle_c - angle_a;
  double allow_fire_ang_max = std::max(angle_a, angle_b) - angle_c;
  double allow_fire_ang_min = std::min(angle_a, angle_b) - angle_c;
  allow_fire_ang_max = tools::limit_rad(allow_fire_ang_max);
  allow_fire_ang_min = tools::limit_rad(allow_fire_ang_min);
  


  // yaw_ang_ref
  double control_delta_angle =
      tools::limit_rad(atan2(target_armor_xyza(1), target_armor_xyza(0)) - gimbal_yaw );
  suggest_fire = (control_delta_angle < allow_fire_ang_max &&
                  control_delta_angle > allow_fire_ang_min);
  if(suggest_fire){
    // tools::logger()->info("fire! control_delta_angle: {},  allow_fire_ang_max: {}, allow_fire_ang_min: {}",
    //   control_delta_angle, allow_fire_ang_max, allow_fire_ang_min
    // );
  }
    
    return suggest_fire;
}

Plan Planner::rbplan(Target target, double bullet_speed, double gimbal_yaw)
{
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
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  Eigen::Vector2d yaw_pitch;
  try {
    yaw_pitch = aim(target, bullet_speed);
    yaw0 = yaw_pitch(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
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

  
  // plan.fire =
  //   std::hypot(
  //     traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
  //     traj(2, HALF_HORIZON + shoot_offset_) -
  //       pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  target.predict(-gimbal_control_delay);
  plan.fire = rbShoot(target, (gimbal_yaw )/57.3 - yaw_offset_);
  // tools::logger()->warn("fire:{}", plan.fire);
  plan.target_yaw = (aim_target_yaw + yaw_offset_ )* 57.3;
  return plan;
}


Plan Planner::rbHeroplan(Target target, double bullet_speed, double gimbal_yaw){
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
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  Eigen::Vector2d yaw_pitch;
  try {
    yaw_pitch = heroaim(target, bullet_speed, gimbal_yaw);
    yaw0 = yaw_pitch(0);
    // traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }



  Plan plan;
  plan.control = true;
  // plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  
  plan.target_pitch = traj(2, HALF_HORIZON);

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
  plan.fire = rbShoot(target, (gimbal_yaw )/57.3 - yaw_offset_);
  // tools::logger()->warn("fire:{}", plan.fire);
  plan.target_yaw = (aim_target_yaw + yaw_offset_ )* 57.3;
  return plan;
}

// Plan Planner::plan(std::optional<Target> target, double bullet_speed, double gimbal_yaw)
// {
//   if (!target.has_value()) return {false};

//   double delay_time =
//     std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

//   auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

//   target->predict(future);

//   // return plan(*target, bullet_speed);
//   return rbplan(*target, bullet_speed, gimbal_yaw);
// }

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

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
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

Eigen::Matrix<double, 2, 1> Planner::heroaim(const Target & target, double bullet_speed, double gimbal_yaw)
{
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

  auto r = target.ekf_x()(8);
  auto v_yaw = target.ekf_x()(7);

    // 旋转中心的坐标
  auto center_x = target.ekf_x()(0);
  auto center_y = target.ekf_x()(2);

  auto aim_point_x = center_x - r*std::cos(gimbal_yaw);
  auto aim_point_y = center_y - r*std::sin(gimbal_yaw);
  auto aim_point_z = xyz.z();

  if(abs(v_yaw) < 0.6){
    aim_point_x = xyz.x();
    aim_point_y = xyz.y();
  }
  auto min_dist1 = min_dist;
  if(target.name == ArmorName::outpost){
    Target target_pitch = target;
    // double pitch_prediction_time_ = 0.05;
    min_dist1 = 1e10;
    Eigen::Vector3d xyz1;
    double yaw1;
    target_pitch.predict(tower_pitch_prediction_time_);
    for (auto & xyza : target.armor_xyza_list()) {
      auto dist = xyza.head<2>().norm();
      if (dist < min_dist1) {
        min_dist1 = dist;
        xyz1 = xyza.head<3>();
        yaw1 = xyza[3];
      }
    }
    aim_point_z = xyz1.z();
  }

  //补偿距离和补偿高度
  double comp_dist = 0;
  double comp_h = 0;
  min_dist = sqrt(aim_point_x*aim_point_x + aim_point_y*aim_point_y);

  debug_xyza = Eigen::Vector4d(aim_point_x, aim_point_y, aim_point_z, yaw);

  auto azim = std::atan2(aim_point_y, aim_point_x);
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist1 - comp_dist, aim_point_z - comp_h);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), bullet_traj.pitch + pitch_offset_};
}


Trajectory Planner::get_trajectory(Target  target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
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

}  // namespace auto_aim