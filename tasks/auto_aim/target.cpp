#include "target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{

double initial_radius(ArmorName name)
{
  if (name == ArmorName::outpost) return motion_model::OUTPOST_RADIUS;
  if (name == ArmorName::base) return 0;
  return 0.26;
}

Eigen::Matrix3d rotation_z(double yaw)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

motion_model::UVLVector uvl_observation(
  const cv::Point2f & top, const cv::Point2f & bottom)
{
  motion_model::UVLVector observation;
  motion_model::UVLMeasure::points_to_observation(
    Eigen::Vector2d(top.x, top.y), Eigen::Vector2d(bottom.x, bottom.y), observation.data());
  return observation;
}

bool finite_pose(const Eigen::Isometry3d & pose)
{
  return pose.matrix().allFinite();
}

bool valid_armor_observation(const Armor & armor)
{
  if (armor.points.size() != 4) return false;
  if (!std::all_of(armor.points.begin(), armor.points.end(), [](const cv::Point2f & point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      })) {
    return false;
  }
  return cv::norm(armor.points[0] - armor.points[3]) > 1e-3 &&
         cv::norm(armor.points[1] - armor.points[2]) > 1e-3;
}

}  // namespace

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point time,
  const EstimatorConfig & config)
: name(armor.name), armor_type(armor.type), priority(armor.priority), config_(config)
{
  initialize(armor, time, initial_radius(name));
}

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point time, double radius,
  int, Eigen::VectorXd initial_covariance_diagonal)
: name(armor.name), armor_type(armor.type), priority(armor.priority)
{
  initialize(armor, time, radius, initial_covariance_diagonal);
}

Target::Target(double x, double vyaw, double radius, double height)
: name(ArmorName::one), armor_type(ArmorType::small), priority(ArmorPriority::fifth)
{
  motion_model::Covariance initial_covariance = motion_model::Covariance::Zero();
  const ArmorName model_name = name;
  const auto inject = [model_name](const auto & delta, auto & nominal) {
    motion_model::inject_state(delta, nominal, model_name);
  };
  const auto box_minus = [model_name](const auto & nominal, const auto & value, auto & delta) {
    motion_model::box_minus_state(nominal, value, delta, model_name);
  };
  estimator_.emplace(inject, box_minus, initial_covariance);
  motion_model::StateVector state = motion_model::StateVector::Zero();
  state[motion_model::index::CX] = x;
  state[motion_model::index::VYAW] = vyaw;
  state[motion_model::index::LOG_R1] = std::log(std::clamp(radius, motion_model::MIN_RADIUS, motion_model::MAX_RADIUS));
  state[motion_model::index::LOG_R2] = state[motion_model::index::LOG_R1];
  state[motion_model::index::H] = height;
  motion_model::clamp_state(state.data(), name);
  estimator_->set_state(state);
  estimator_->set_iteration_num(config_.iterations);
  initialized_ = true;
  observed_ids_.assign(armor_count(), false);
  observed_ids_[0] = true;
  voter_.reset(time_, yaw());
}

void Target::initialize(
  const Armor & armor, std::chrono::steady_clock::time_point time, double radius,
  const std::optional<Eigen::VectorXd> & covariance_diagonal)
{
  if (!armor.pnp_valid || !finite_pose(armor.pose_in_world)) {
    throw std::invalid_argument("Target initialization requires a valid armor pose");
  }

  motion_model::Covariance initial_covariance = motion_model::Covariance::Zero();
  initial_covariance.diagonal() << 1, 10, 1, 10, 1, 10, 1, 100, 1, 1, 1, 1, 1;
  if (covariance_diagonal && covariance_diagonal->size() == motion_model::STATE_DIM) {
    initial_covariance.diagonal() = *covariance_diagonal;
  }

  const ArmorName model_name = name;
  const auto inject = [model_name](const auto & delta, auto & nominal) {
    motion_model::inject_state(delta, nominal, model_name);
  };
  const auto box_minus = [model_name](const auto & nominal, const auto & value, auto & delta) {
    motion_model::box_minus_state(nominal, value, delta, model_name);
  };
  estimator_.emplace(inject, box_minus, initial_covariance);
  estimator_->set_iteration_num(config_.iterations);

  const double model_radius = (name == ArmorName::base) ? 0 : radius;
  Eigen::Isometry3d armor_in_car = Eigen::Isometry3d::Identity();
  armor_in_car.translation() << -model_radius, 0, 0;
  const double pitch =
    name == ArmorName::outpost ? motion_model::OUTPOST_ARMOR_PITCH
                               : motion_model::NORMAL_ARMOR_PITCH;
  armor_in_car.linear() =
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
  Eigen::Isometry3d car_in_world = armor.pose_in_world * armor_in_car.inverse();
  if (motion_model::yaw_only(name)) {
    const double car_yaw = std::atan2(car_in_world.linear()(1, 0), car_in_world.linear()(0, 0));
    car_in_world.linear() = rotation_z(car_yaw);
  }

  motion_model::StateVector state = motion_model::StateVector::Zero();
  state[motion_model::index::CX] = car_in_world.translation().x();
  state[motion_model::index::CY] = car_in_world.translation().y();
  state[motion_model::index::CZ] = car_in_world.translation().z();
  const Eigen::Matrix3d car_rotation = car_in_world.linear();
  const Eigen::Vector3d rotation_vector = tools::so3_log(car_rotation);
  state[motion_model::index::ROT_X] = rotation_vector.x();
  state[motion_model::index::ROT_Y] = rotation_vector.y();
  state[motion_model::index::ROT_Z] = rotation_vector.z();
  const double stored_radius = name == ArmorName::outpost ? motion_model::OUTPOST_RADIUS
                                                          : std::max(radius, 0.26);
  state[motion_model::index::LOG_R1] = std::log(stored_radius);
  state[motion_model::index::LOG_R2] = std::log(stored_radius);
  motion_model::clamp_state(state.data(), name);
  estimator_->set_state(state);

  time_ = time;
  xyz_in_world = armor.xyz_in_world;
  observed_ids_.assign(armor_count(), false);
  observed_ids_[0] = true;
  initialized_ = true;
  jumped = false;
  last_id = 0;
  update_count_ = 0;
  voter_.reset(time, yaw());
}

void Target::predict(std::chrono::steady_clock::time_point time)
{
  if (!initialized_ || !estimator_) return;
  const double dt = tools::delta_time(time, time_);
  if (!std::isfinite(dt) || dt < 0) return;
  const auto Q = process_noise(dt);
  const motion_model::Predict predict_func{dt, name, voter_.state};
  estimator_->predict(predict_func, Q);
  clamp_nominal_state();
  time_ = time;
}

void Target::predict(double dt)
{
  if (!initialized_ || !estimator_ || !std::isfinite(dt) || dt < 0) return;
  const auto Q = process_noise(dt);
  const motion_model::Predict predict_func{dt, name, voter_.state};
  estimator_->predict(predict_func, Q);
  clamp_nominal_state();
  time_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(dt));
}

motion_model::Covariance Target::process_noise(double dt) const
{
  motion_model::Covariance noise = motion_model::Covariance::Zero();
  const Eigen::Vector3d acceleration =
    name == ArmorName::outpost ? config_.outpost_acceleration : config_.common_acceleration;
  const double yaw_acceleration = name == ArmorName::outpost
                                    ? config_.outpost_yaw_acceleration
                                    : config_.common_yaw_acceleration;
  const Eigen::Matrix3d world_acceleration =
    rotation() * acceleration.asDiagonal() * rotation().transpose();
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  constexpr std::array<int, 3> position_indices{
    motion_model::index::CX, motion_model::index::CY, motion_model::index::CZ};
  constexpr std::array<int, 3> velocity_indices{
    motion_model::index::VCX, motion_model::index::VCY, motion_model::index::VCZ};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      noise(position_indices[row], position_indices[column]) =
        0.25 * dt4 * world_acceleration(row, column);
      noise(position_indices[row], velocity_indices[column]) =
        0.5 * dt3 * world_acceleration(row, column);
      noise(velocity_indices[row], position_indices[column]) =
        0.5 * dt3 * world_acceleration(row, column);
      noise(velocity_indices[row], velocity_indices[column]) =
        dt2 * world_acceleration(row, column);
    }
  }

  noise(motion_model::index::ROT_Z, motion_model::index::ROT_Z) +=
    0.25 * dt4 * yaw_acceleration;
  noise(motion_model::index::ROT_Z, motion_model::index::VYAW) +=
    0.5 * dt3 * yaw_acceleration;
  noise(motion_model::index::VYAW, motion_model::index::ROT_Z) +=
    0.5 * dt3 * yaw_acceleration;
  noise(motion_model::index::VYAW, motion_model::index::VYAW) +=
    dt2 * yaw_acceleration;

  if (!motion_model::yaw_only(name)) {
    noise(motion_model::index::ROT_X, motion_model::index::ROT_X) +=
      dt * config_.roll_pitch_random_walk;
    noise(motion_model::index::ROT_Y, motion_model::index::ROT_Y) +=
      dt * config_.roll_pitch_random_walk;
  }
  if (name == ArmorName::outpost) {
    noise(motion_model::index::OUTPOST_DZ1, motion_model::index::OUTPOST_DZ1) =
      config_.outpost_height_random_walk;
    noise(motion_model::index::OUTPOST_DZ2, motion_model::index::OUTPOST_DZ2) =
      config_.outpost_height_random_walk;
  } else if (name != ArmorName::base) {
    const double radius1 = radius(0);
    const double radius2 = radius(1);
    noise(motion_model::index::LOG_R1, motion_model::index::LOG_R1) =
      config_.radius_random_walk / (radius1 * radius1);
    noise(motion_model::index::LOG_R2, motion_model::index::LOG_R2) =
      config_.radius_random_walk / (radius2 * radius2);
    noise(motion_model::index::H, motion_model::index::H) = config_.height_random_walk;
  }
  return noise;
}

int Target::update(
  const std::vector<std::pair<int, Armor>> & matched_armors, int primary_id,
  const CameraContext & camera, std::chrono::steady_clock::time_point time)
{
  if (!initialized_ || !estimator_ || matched_armors.empty()) return 0;
  std::vector<std::shared_ptr<motion_model::RobotESEKF::ObservationBase>> observations;
  observations.reserve(matched_armors.size() * 2 + 1);
  int valid_armor_count = 0;

  for (const auto & match : matched_armors) {
    const int id = match.first;
    const Armor & armor = match.second;
    if (id < 0 || id >= armor_count() || !valid_armor_observation(armor)) continue;
    ++valid_armor_count;
    for (const bool left : {true, false}) {
      const int top_index = left ? 0 : 1;
      const int bottom_index = left ? 3 : 2;
      const cv::Point2f top = armor.points[top_index];
      const cv::Point2f bottom = armor.points[bottom_index];
      const auto measured = uvl_observation(top, bottom);
      const double length = cv::norm(top - bottom);
      const double pixel_sigma = config_.uvl_pixel_sigma_ratio * length;
      const double length_sigma = config_.uvl_length_sigma_ratio * length;
      const double angle_sigma = config_.uvl_angle_sigma;
      motion_model::UVLContext context{id, left, name, armor_type, camera};
      motion_model::UVLMeasure measure{context};
      const auto noise = [pixel_sigma, length_sigma, angle_sigma](const auto &) {
        Eigen::Matrix<double, motion_model::UVL_DIM, motion_model::UVL_DIM> R =
          Eigen::Matrix<double, motion_model::UVL_DIM, motion_model::UVL_DIM>::Zero();
        R(motion_model::index::UVL_ANGLE, motion_model::index::UVL_ANGLE) =
          angle_sigma * angle_sigma / 2;
        R(motion_model::index::UVL_CENTER_X, motion_model::index::UVL_CENTER_X) =
          pixel_sigma * pixel_sigma / 2;
        R(motion_model::index::UVL_CENTER_Y, motion_model::index::UVL_CENTER_Y) =
          pixel_sigma * pixel_sigma / 2;
        R(motion_model::index::UVL_LENGTH, motion_model::index::UVL_LENGTH) =
          length_sigma * length_sigma / 2;
        return R;
      };
      observations.push_back(estimator_->make_observation(
        measured, measure, noise,
        [](const auto & predicted, const auto & observation) {
          return motion_model::UVLMeasure::residual(predicted, observation);
        }));
    }
  }

  if (
    matched_armors.size() == 1 && valid_armor_count == 1 &&
    matched_armors.front().second.pnp_valid) {
    const int id = matched_armors.front().first;
    const Armor & armor = matched_armors.front().second;
    const Eigen::Isometry3d pose_in_camera = camera.T_camera_world.inverse() * armor.pose_in_world;
    const auto points = armor_object_points(armor_type);
    const Eigen::Vector3d left_center = (pose_in_camera * points[0] + pose_in_camera * points[3]) / 2;
    const Eigen::Vector3d right_center = (pose_in_camera * points[1] + pose_in_camera * points[2]) / 2;
    motion_model::DiffVector measured;
    measured[0] = left_center.z() - right_center.z();
    motion_model::DiffMeasure measure{{id, true, name, armor_type, camera}};
    const double sigma = config_.depth_difference_sigma;
    observations.push_back(estimator_->make_observation(
      measured, measure,
      [sigma](const auto &) {
        Eigen::Matrix<double, 1, 1> R;
        R(0, 0) = sigma * sigma / 2;
        return R;
      },
      [](const auto & predicted, const auto & observation) {
        return motion_model::DiffMeasure::residual(predicted, observation);
      }));
  }

  if (observations.empty() || !estimator_->update_multi(observations)) return 0;
  clamp_nominal_state();
  for (const auto & match : matched_armors) {
    if (
      match.first < 0 || match.first >= armor_count() ||
      !valid_armor_observation(match.second)) {
      continue;
    }
    if (match.first >= 0 && match.first < static_cast<int>(observed_ids_.size())) {
      observed_ids_[match.first] = true;
      jumped = jumped || match.first != 0;
    }
    if (match.first == primary_id && match.second.pnp_valid) xyz_in_world = match.second.xyz_in_world;
  }
  last_id = primary_id;
  ++update_count_;
  if (name == ArmorName::outpost) voter_.update(yaw(), time);
  return static_cast<int>(observations.size());
}

void Target::clamp_nominal_state()
{
  if (!estimator_) return;
  auto state = estimator_->state();
  motion_model::clamp_state(state.data(), name);
  estimator_->set_state(state);
}

Eigen::VectorXd Target::ekf_x() const
{
  if (!estimator_) return Eigen::VectorXd::Zero(motion_model::STATE_DIM);
  return estimator_->state();
}

const motion_model::Covariance & Target::covariance() const
{
  static const motion_model::Covariance empty = motion_model::Covariance::Zero();
  return estimator_ ? estimator_->covariance() : empty;
}

tools::EstimatorDiagnostics Target::estimator_diagnostics() const
{
  return estimator_ ? estimator_->diagnostics() : tools::EstimatorDiagnostics{};
}

Eigen::Vector3d Target::center() const
{
  if (!estimator_) return Eigen::Vector3d::Zero();
  const auto & state = estimator_->state();
  return {state[motion_model::index::CX], state[motion_model::index::CY], state[motion_model::index::CZ]};
}

Eigen::Vector3d Target::velocity() const
{
  if (!estimator_) return Eigen::Vector3d::Zero();
  const auto & state = estimator_->state();
  return {state[motion_model::index::VCX], state[motion_model::index::VCY], state[motion_model::index::VCZ]};
}

Eigen::Matrix3d Target::rotation() const
{
  if (!estimator_) return Eigen::Matrix3d::Identity();
  return motion_model::state_rotation(estimator_->state(), name);
}

double Target::yaw() const
{
  const auto R = rotation();
  return std::atan2(R(1, 0), R(0, 0));
}

double Target::yaw_rate() const
{
  return estimator_ ? estimator_->state()[motion_model::index::VYAW] : 0;
}

double Target::radius(int id) const
{
  if (!estimator_ || id < 0 || id >= armor_count()) return 0;
  return motion_model::armor_radius(estimator_->state().data(), id, name);
}

double Target::armor_height(int id) const
{
  const auto poses = armor_pose_list();
  return id >= 0 && id < static_cast<int>(poses.size()) ? poses[id].translation().z() : center().z();
}

int Target::armor_count() const { return motion_model::armor_count(name); }

std::vector<Eigen::Isometry3d> Target::armor_pose_list() const
{
  std::vector<Eigen::Isometry3d> poses;
  if (!estimator_) return poses;
  poses.reserve(armor_count());
  for (int id = 0; id < armor_count(); ++id) {
    poses.push_back(motion_model::armor_pose(estimator_->state().data(), id, name));
  }
  return poses;
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> result;
  for (const auto & pose : armor_pose_list()) {
    const double armor_yaw = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
    result.emplace_back(
      pose.translation().x(), pose.translation().y(), pose.translation().z(), armor_yaw);
  }
  return result;
}

bool Target::matching_initialized() const
{
  if (name == ArmorName::base) return true;
  if (name != ArmorName::outpost) return jumped;
  return !observed_ids_.empty() &&
         std::all_of(observed_ids_.begin(), observed_ids_.end(), [](bool observed) { return observed; });
}

bool Target::diverged() const
{
  if (!initialized_ || !estimator_ || !estimator_->state().allFinite() ||
      !estimator_->covariance().allFinite())
    return true;
  for (const auto & pose : armor_pose_list()) {
    if (!finite_pose(pose)) return true;
  }
  const auto & state = estimator_->state();
  if (name == ArmorName::outpost) {
    return std::abs(state[motion_model::index::OUTPOST_DZ1]) > 0.300001 ||
           std::abs(state[motion_model::index::OUTPOST_DZ2]) > 0.300001;
  }
  if (name == ArmorName::base) return false;
  return radius(0) < motion_model::MIN_RADIUS || radius(0) > motion_model::MAX_RADIUS ||
         radius(1) < motion_model::MIN_RADIUS || radius(1) > motion_model::MAX_RADIUS ||
         std::abs(state[motion_model::index::H]) > 0.500001;
}

bool Target::convergened()
{
  const int required_updates = name == ArmorName::outpost ? 10 : 3;
  if (update_count_ > required_updates && !diverged()) converged_ = true;
  return converged_;
}

}  // namespace auto_aim
