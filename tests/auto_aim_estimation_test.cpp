#include <Eigen/Eigenvalues>

#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/motion_model.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/math_tools.hpp"

namespace
{
using auto_aim::Armor;
using auto_aim::ArmorName;
using auto_aim::ArmorType;
using auto_aim::CameraContext;
using auto_aim::Solver;
using auto_aim::Target;
namespace model = auto_aim::motion_model;

constexpr double PI = 3.14159265358979323846;

struct Runner
{
  int failures = 0;

  void check(bool condition, const std::string & message)
  {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
};

Eigen::Matrix3d front_facing_rotation()
{
  Eigen::Matrix3d rotation;
  rotation << 0, -1, 0, 0, 0, -1, 1, 0, 0;
  return Eigen::AngleAxisd(0.16, Eigen::Vector3d::UnitY()).toRotationMatrix() * rotation;
}

Eigen::Isometry3d synthetic_pose_in_world(const CameraContext & camera)
{
  Eigen::Isometry3d pose_in_camera = Eigen::Isometry3d::Identity();
  pose_in_camera.linear() = front_facing_rotation();
  pose_in_camera.translation() << 0.08, -0.04, 3.2;
  return camera.T_camera_world * pose_in_camera;
}

Armor armor_from_pose(
  const Solver & solver, const Eigen::Isometry3d & pose, ArmorName name = ArmorName::three,
  ArmorType type = ArmorType::small)
{
  Armor armor;
  armor.color = auto_aim::Color::blue;
  armor.name = name;
  armor.type = type;
  armor.pose_in_world = pose;
  armor.xyz_in_world = pose.translation();
  armor.pnp_valid = true;
  armor.points = solver.reproject_pose(pose, type);
  if (armor.points.size() == 4) {
    armor.center = (armor.points[0] + armor.points[1] + armor.points[2] + armor.points[3]) / 4;
  }
  return armor;
}

void test_so3_and_injection(Runner & runner)
{
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  runner.check(
    (tools::so3_exp(zero) - Eigen::Matrix3d::Identity()).norm() < 1e-14,
    "SO(3) exponential is stable at zero");

  const Eigen::Vector3d near_pi((PI - 1e-8), 0, 0);
  const Eigen::Matrix3d near_pi_rotation = tools::so3_exp(near_pi);
  runner.check(
    (tools::so3_exp(tools::so3_log(near_pi_rotation)) - near_pi_rotation).norm() < 1e-8,
    "SO(3) logarithm is stable near pi");

  std::mt19937 random(20260720);
  std::normal_distribution<double> perturbation(0.0, 1e-4);
  for (int sample = 0; sample < 100; ++sample) {
    model::StateVector state = model::StateVector::Zero();
    state[model::index::ROT_X] = 0.2;
    state[model::index::ROT_Y] = -0.1;
    state[model::index::ROT_Z] = 0.3;
    state[model::index::LOG_R1] = std::log(0.26);
    state[model::index::LOG_R2] = std::log(0.28);
    model::StateVector delta;
    for (int i = 0; i < model::STATE_DIM; ++i) delta[i] = perturbation(random);
    model::StateVector injected = state;
    model::inject_state(delta, injected, ArmorName::three);
    model::StateVector recovered;
    model::box_minus_state(state, injected, recovered, ArmorName::three);
    runner.check(
      (recovered - delta).cwiseAbs().maxCoeff() < 1e-9,
      "box_minus(x, inject(delta, x)) recovers a small SO(3) error");
  }

  model::StateVector state = model::StateVector::Zero();
  state[model::index::ROT_Z] = PI - 1e-4;
  state[model::index::VYAW] = 1.0;
  state[model::index::LOG_R1] = std::log(0.26);
  state[model::index::LOG_R2] = std::log(0.26);
  model::StateVector predicted;
  model::Predict{0.01, ArmorName::three, model::DirectionVoter::collecting}(
    state.data(), predicted.data());
  const Eigen::Matrix3d expected =
    model::state_rotation(state, ArmorName::three) *
    tools::so3_exp(Eigen::Vector3d(0, 0, 0.01));
  runner.check(
    (model::state_rotation(predicted, ArmorName::three) - expected).norm() < 1e-10,
    "SO(3) prediction remains continuous while crossing pi");

  model::DirectionVoter voter;
  const auto start = std::chrono::steady_clock::time_point{};
  voter.reset(start, 0);
  for (int vote = 1; vote <= 11; ++vote) {
    voter.update(0.1 * vote, start + std::chrono::milliseconds(1000 + vote));
  }
  runner.check(voter.state == model::DirectionVoter::clockwise, "outpost direction vote stabilizes");
  state[model::index::VYAW] = 0.2;
  model::Predict{0.01, ArmorName::outpost, voter.state}(state.data(), predicted.data());
  runner.check(
    std::abs(predicted[model::index::VYAW] - model::OUTPOST_YAW_RATE) < 1e-12,
    "stable outpost direction fixes yaw rate to 2.51 rad/s");
  state[model::index::VYAW] = 4;
  model::Predict{0.01, ArmorName::base, model::DirectionVoter::collecting}(
    state.data(), predicted.data());
  runner.check(predicted[model::index::VYAW] == 0, "base prediction forces zero yaw rate");

  const Eigen::Vector3d invalid_projection = {0, 0, 0};
  const auto invalid_pixel = tools::project_point(
    invalid_projection, Eigen::Matrix3d::Identity(), Eigen::Matrix<double, 5, 1>::Zero());
  runner.check(!invalid_pixel.allFinite(), "non-positive camera depth is rejected");
}

void test_geometry_and_clamps(Runner & runner)
{
  model::StateVector state = model::StateVector::Zero();
  state[model::index::LOG_R1] = std::log(0.26);
  state[model::index::LOG_R2] = std::log(0.31);
  for (const auto & item : std::vector<std::pair<ArmorName, int>>{
         {ArmorName::three, 4}, {ArmorName::outpost, 3}, {ArmorName::base, 1}}) {
    std::vector<Eigen::Isometry3d> poses;
    for (int id = 0; id < model::armor_count(item.first); ++id) {
      poses.push_back(model::armor_pose(state.data(), id, item.first));
    }
    runner.check(
      static_cast<int>(poses.size()) == item.second,
      "normal, outpost and base models generate 4/3/1 armor poses");
  }
  const auto base_pose = model::armor_pose(state.data(), 0, ArmorName::base);
  runner.check(base_pose.translation().norm() < 1e-14, "base armor is located at model center");

  state[model::index::LOG_R1] = std::log(1e-3);
  state[model::index::LOG_R2] = std::log(5.0);
  state[model::index::H] = 2.0;
  model::clamp_state(state.data(), ArmorName::three);
  runner.check(
    std::abs(std::exp(state[model::index::LOG_R1]) - model::MIN_RADIUS) < 1e-12 &&
      std::abs(std::exp(state[model::index::LOG_R2]) - model::MAX_RADIUS) < 1e-12,
    "log radii clamp to positive physical bounds");
  runner.check(std::abs(state[model::index::H] - 0.5) < 1e-12, "normal height clamps to 0.5 m");

  state[model::index::OUTPOST_DZ1] = -1;
  state[model::index::OUTPOST_DZ2] = 1;
  model::clamp_state(state.data(), ArmorName::outpost);
  runner.check(
    std::abs(state[model::index::OUTPOST_DZ1] + 0.3) < 1e-12 &&
      std::abs(state[model::index::OUTPOST_DZ2] - 0.3) < 1e-12,
    "outpost height slots clamp to 0.3 m");
  runner.check(
    std::abs(model::armor_radius(state.data(), 0, ArmorName::outpost) - model::OUTPOST_RADIUS) <
      1e-12,
    "outpost radius remains fixed");
}

void test_camera_and_pnp(Runner & runner, Solver & solver)
{
  const auto yaml = YAML::LoadFile(std::string(SP_VISION_SOURCE_DIR) + "/configs/rb_auto_aim.yaml");
  const auto rotation_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  const auto translation_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  const auto gimbal_body_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  const Eigen::Matrix3d camera_to_gimbal =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation_data.data());
  const Eigen::Vector3d camera_translation = Eigen::Map<const Eigen::Vector3d>(translation_data.data());
  const Eigen::Matrix3d gimbal_to_body =
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(gimbal_body_data.data());
  const Eigen::Quaterniond body_rotation(
    Eigen::AngleAxisd(0.31, Eigen::Vector3d(0.2, -0.4, 0.7).normalized()));
  solver.set_R_gimbal2world(body_rotation);
  const Eigen::Matrix3d gimbal_to_world =
    gimbal_to_body.transpose() * body_rotation.toRotationMatrix() * gimbal_to_body;
  const auto context = solver.camera_context();
  runner.check(
    context.camera_matrix.allFinite() && context.distortion.allFinite(),
    "CameraContext returns finite intrinsic values");
  runner.check(
    context.T_camera_world.linear().isApprox(gimbal_to_world * camera_to_gimbal, 1e-12) &&
      context.T_camera_world.translation().isApprox(gimbal_to_world * camera_translation, 1e-12),
    "CameraContext composes camera-to-world rotation and translation exactly");
  const auto pose = synthetic_pose_in_world(context);
  Armor armor = armor_from_pose(solver, pose, ArmorName::outpost, ArmorType::small);
  runner.check(armor.points.size() == 4, "complete synthetic armor pose projects to four corners");
  armor.pnp_valid = false;
  runner.check(solver.solve(armor), "IPPE finds a front-facing synthetic pose");
  runner.check(armor.pnp_valid && armor.pose_in_world.matrix().allFinite(), "IPPE stores full pose");
  const auto solved_in_camera = context.T_camera_world.inverse() * armor.pose_in_world;
  runner.check(solved_in_camera.translation().z() > 0, "IPPE candidate has positive depth");
  runner.check(
    (-solved_in_camera.linear().col(0)).dot(-solved_in_camera.translation()) > 0,
    "IPPE candidate faces the camera");
  const double solved_yaw = std::atan2(armor.pose_in_world.linear()(1, 0), armor.pose_in_world.linear()(0, 0));
  const Eigen::Matrix3d yaw_only =
    Eigen::AngleAxisd(solved_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  runner.check(
    (armor.pose_in_world.linear() - yaw_only).norm() > 1e-3,
    "Solver preserves full IPPE pose even for a yaw-only target class");
  const auto reprojection = solver.reproject_pose(armor.pose_in_world, armor.type);
  double reprojection_error = 0;
  for (int i = 0; i < 4 && reprojection.size() == 4; ++i) {
    reprojection_error += cv::norm(reprojection[i] - armor.points[i]);
  }
  runner.check(reprojection.size() == 4 && reprojection_error < 1e-2, "IPPE pose reprojects correctly");

  armor.points.resize(3);
  armor.xyz_in_world.setOnes();
  runner.check(!solver.solve(armor), "invalid corner count fails PnP");
  runner.check(
    !armor.pnp_valid && armor.xyz_in_world.isZero(0) &&
      armor.pose_in_world.matrix().isApprox(Eigen::Isometry3d::Identity().matrix()),
    "failed PnP clears stale pose fields");
  armor.points.assign(4, cv::Point2f(20, 20));
  runner.check(!solver.solve(armor), "degenerate four-corner geometry fails PnP cleanly");
}

Target make_target(const Solver & solver, ArmorName name, ArmorType type)
{
  Armor armor = armor_from_pose(solver, synthetic_pose_in_world(solver.camera_context()), name, type);
  return Target(armor, std::chrono::steady_clock::time_point{});
}

void test_target_models_and_copy(Runner & runner, const Solver & solver)
{
  Target normal = make_target(solver, ArmorName::three, ArmorType::small);
  Target outpost = make_target(solver, ArmorName::outpost, ArmorType::small);
  Target base = make_target(solver, ArmorName::base, ArmorType::big);
  runner.check(
    normal.armor_pose_list().size() == 4 && outpost.armor_pose_list().size() == 3 &&
      base.armor_pose_list().size() == 1,
    "Target exposes 4/3/1 armor poses");
  runner.check(
    std::abs(outpost.radius(0) - model::OUTPOST_RADIUS) < 1e-12 && base.radius(0) == 0,
    "Target exposes fixed outpost and zero base radii");
  runner.check(
    std::abs(outpost.rotation()(2, 0)) < 1e-12 && std::abs(outpost.rotation()(2, 1)) < 1e-12,
    "outpost initialization constrains the car to yaw-only");
  runner.check(
    (base.center() - synthetic_pose_in_world(solver.camera_context()).translation()).norm() < 1e-9,
    "base center initializes at armor center");

  Target original(2.0, 0.8, 0.26, 0.1);
  const auto original_state = original.ekf_x();
  const auto original_covariance = original.covariance();
  const auto original_time = original.getTimePoint();
  Target copy = original;
  copy.predict(0.25);
  runner.check(original.ekf_x().isApprox(original_state, 0), "copy prediction does not alter original state");
  runner.check(
    original.covariance().isApprox(original_covariance, 0),
    "copy prediction does not alter original covariance");
  runner.check(original.getTimePoint() == original_time, "copy prediction does not alter original timestamp");
  runner.check(!copy.ekf_x().isApprox(original_state, 0), "copied Target predicts independently");
}

Armor measurement_for_pose(const Solver & solver, const Eigen::Isometry3d & pose)
{
  Armor armor = armor_from_pose(solver, pose, ArmorName::three, ArmorType::small);
  armor.pnp_valid = false;
  return armor;
}

void translate_measurement(Armor & armor, const cv::Point2f & offset)
{
  for (auto & point : armor.points) point += offset;
  armor.center += offset;
  armor.pnp_valid = false;
}

void test_observations_and_filter(Runner & runner, Solver & solver)
{
  Target target = make_target(solver, ArmorName::three, ArmorType::small);
  const auto context = solver.camera_context();
  const auto state_dynamic = target.ekf_x();
  model::StateVector state = state_dynamic;

  model::UVLMeasure left{{0, true, target.name, target.armor_type, context}};
  model::UVLVector predicted_left;
  left(state.data(), predicted_left.data());
  runner.check(
    predicted_left.allFinite() &&
      model::UVLMeasure::residual(predicted_left, predicted_left).norm() < 1e-12,
    "synthetic UVL projection has zero residual");

  model::DiffMeasure difference{{0, true, target.name, target.armor_type, context}};
  model::DiffVector predicted_difference;
  difference(state.data(), predicted_difference.data());
  const auto pose_in_camera = context.T_camera_world.inverse() * target.armor_pose_list()[0];
  const auto object_points = auto_aim::armor_object_points(target.armor_type);
  const Eigen::Vector3d left_center =
    (pose_in_camera * object_points[0] + pose_in_camera * object_points[3]) / 2;
  const Eigen::Vector3d right_center =
    (pose_in_camera * object_points[1] + pose_in_camera * object_points[2]) / 2;
  runner.check(
    std::abs(predicted_difference[0] - (left_center.z() - right_center.z())) < 1e-12 &&
      std::abs(predicted_difference[0]) > 1e-5,
    "DiffMeasure uses the same signed left-minus-right depth convention as PnP (predicted=" +
      std::to_string(predicted_difference[0]) + ", direct=" +
      std::to_string(left_center.z() - right_center.z()) + ")");

  Armor single = measurement_for_pose(solver, target.armor_pose_list()[0]);
  runner.check(solver.solve(single), "single-board synthetic observation has a valid IPPE pose");
  const auto solved_pose_in_camera = context.T_camera_world.inverse() * single.pose_in_world;
  const Eigen::Vector3d solved_left_center =
    (solved_pose_in_camera * object_points[0] + solved_pose_in_camera * object_points[3]) / 2;
  const Eigen::Vector3d solved_right_center =
    (solved_pose_in_camera * object_points[1] + solved_pose_in_camera * object_points[2]) / 2;
  runner.check(
    predicted_difference[0] * (solved_left_center.z() - solved_right_center.z()) > 0,
    "DiffMeasure sign agrees with the selected IPPE pose");
  const int single_blocks = target.update(
    {{0, single}}, 0, context, std::chrono::steady_clock::time_point{});
  const auto single_diagnostics = target.estimator_diagnostics();
  runner.check(single_blocks == 3, "single board contributes two UVL blocks and one Diff block");
  runner.check(single_diagnostics.observation_dim == 9, "single board joint observation is 9D");

  Target two_board_target = make_target(solver, ArmorName::three, ArmorType::small);
  const auto poses = two_board_target.armor_pose_list();
  Armor first = measurement_for_pose(solver, poses[0]);
  Armor second = measurement_for_pose(solver, poses[1]);
  const int double_blocks = two_board_target.update(
    {{0, first}, {1, second}}, 1, context, std::chrono::steady_clock::time_point{});
  const auto double_diagnostics = two_board_target.estimator_diagnostics();
  runner.check(double_blocks == 4, "two boards contribute four UVL blocks without Diff");
  runner.check(double_diagnostics.observation_dim == 16, "two-board joint observation is 16D");
  runner.check(two_board_target.last_id == 1, "explicit primary match deterministically sets last_id");
  runner.check(two_board_target.matching_initialized(), "observing a new normal armor enables normal gate");
  runner.check(
    two_board_target.ekf_x().allFinite() && two_board_target.covariance().allFinite(),
    "multi-observation update keeps state and covariance finite");
  runner.check(
    two_board_target.covariance().isApprox(two_board_target.covariance().transpose(), 1e-10),
    "posterior covariance remains symmetric");
  Eigen::SelfAdjointEigenSolver<model::Covariance> eigenvalues(two_board_target.covariance());
  runner.check(
    eigenvalues.info() == Eigen::Success && eigenvalues.eigenvalues().minCoeff() > -1e-8,
    "posterior covariance remains positive semidefinite");
}

void test_config_compatibility(Runner & runner)
{
  const std::string root = SP_VISION_SOURCE_DIR;
  try {
    Solver old_solver(root + "/configs/xiaohei.yaml");
    auto_aim::Tracker old_tracker(root + "/configs/xiaohei.yaml", &old_solver);
    Solver new_solver(root + "/configs/rb_auto_aim.yaml");
    auto_aim::Tracker new_tracker(root + "/configs/rb_auto_aim.yaml", &new_solver);
    runner.check(old_tracker.state() == "lost", "legacy YAML without estimator uses defaults");
    runner.check(new_tracker.state() == "lost", "complete estimator YAML loads successfully");
  } catch (const std::exception & error) {
    runner.check(false, std::string("configuration compatibility threw: ") + error.what());
  }
}

void test_tracker_matching(Runner & runner)
{
  const std::string config_path = std::string(SP_VISION_SOURCE_DIR) + "/configs/rb_auto_aim.yaml";
  Solver solver(config_path);
  const auto start = std::chrono::steady_clock::time_point{};
  Armor initial = armor_from_pose(
    solver, synthetic_pose_in_world(solver.camera_context()), ArmorName::three, ArmorType::small);

  auto_aim::Tracker tracker(config_path, &solver);
  Armor invalid = initial;
  invalid.points.assign(4, cv::Point2f(10, 10));
  invalid.priority = auto_aim::ArmorPriority::first;
  initial.priority = auto_aim::ArmorPriority::fifth;
  std::list<Armor> detections{invalid, initial};
  auto targets = tracker.test_track(detections, start);
  runner.check(
    !targets.empty() && tracker.state() == "detecting",
    "tracker skips an invalid leading PnP candidate and initializes from the next valid armor");
  for (int frame = 1; frame <= 4 && !targets.empty(); ++frame) {
    Armor detection = measurement_for_pose(solver, targets.front().armor_pose_list()[0]);
    detections = {detection};
    targets = tracker.test_track(detections, start + std::chrono::milliseconds(frame * 10));
  }
  runner.check(!targets.empty() && tracker.state() == "tracking", "tracker reaches tracking state");

  if (!targets.empty()) {
    const auto poses = targets.front().armor_pose_list();
    Armor id0 = measurement_for_pose(solver, poses[0]);
    translate_measurement(id0, {0.5f, 0});
    Armor id1 = measurement_for_pose(solver, poses[1]);
    detections = {id0, id1};
    targets = tracker.test_track(detections, start + std::chrono::milliseconds(50));
  }
  runner.check(
    !targets.empty() && targets.front().estimator_diagnostics().observation_dim == 16,
    "global greedy matching uses two detections and two predicted IDs once each");
  runner.check(
    !targets.empty() && targets.front().last_id == 1,
    "lowest-cost match determines primary ID independent of detection order");

  if (!targets.empty()) {
    Armor id1 = measurement_for_pose(solver, targets.front().armor_pose_list()[1]);
    detections = {id1};
    targets = tracker.test_track(detections, start + std::chrono::milliseconds(60));
  }
  runner.check(
    !targets.empty() && targets.front().last_id == 1 &&
      targets.front().estimator_diagnostics().observation_dim == 9,
    "armor ID remains continuous after switching to one board");

  if (!targets.empty()) {
    Armor outside_normal_gate = measurement_for_pose(solver, targets.front().armor_pose_list()[1]);
    translate_measurement(outside_normal_gate, {0, 100});
    detections = {outside_normal_gate};
    targets = tracker.test_track(detections, start + std::chrono::milliseconds(70));
  }
  runner.check(
    !targets.empty() && targets.front().estimator_diagnostics().observation_dim == 0 &&
      tracker.state() == "temp_lost",
    "initialized matching uses the normal 200 gate");

  auto_aim::Tracker bootstrap_tracker(config_path, &solver);
  detections = {initial};
  auto bootstrap_targets = bootstrap_tracker.test_track(detections, start);
  if (!bootstrap_targets.empty()) {
    Armor inside_bootstrap_gate =
      measurement_for_pose(solver, bootstrap_targets.front().armor_pose_list()[0]);
    translate_measurement(inside_bootstrap_gate, {0, 100});
    detections = {inside_bootstrap_gate};
    bootstrap_targets =
      bootstrap_tracker.test_track(detections, start + std::chrono::milliseconds(10));
  }
  runner.check(
    !bootstrap_targets.empty() &&
      bootstrap_targets.front().estimator_diagnostics().observation_dim == 9,
    "uninitialized matching uses the wider 1000 gate");
}

}  // namespace

int main()
{
  Runner runner;
  const std::string root = SP_VISION_SOURCE_DIR;
  Solver solver(root + "/configs/rb_auto_aim.yaml");

  test_so3_and_injection(runner);
  test_geometry_and_clamps(runner);
  test_camera_and_pnp(runner, solver);
  test_target_models_and_copy(runner, solver);
  test_observations_and_filter(runner, solver);
  test_config_compatibility(runner);
  test_tracker_matching(runner);

  if (runner.failures != 0) {
    std::cerr << runner.failures << " estimation test(s) failed\n";
    return 1;
  }
  std::cout << "All Auto Aim estimation tests passed\n";
  return 0;
}
