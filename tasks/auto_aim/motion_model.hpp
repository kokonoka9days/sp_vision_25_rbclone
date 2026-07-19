#ifndef AUTO_AIM__MOTION_MODEL_HPP
#define AUTO_AIM__MOTION_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/jet.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

#include "armor.hpp"
#include "solver.hpp"
#include "tools/error_state_extended_kalman_filter.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

struct EstimatorConfig
{
  int iterations = 5;
  Eigen::Vector3d common_acceleration = {30, 30, 1};
  Eigen::Vector3d outpost_acceleration = {1, 1, 1};
  double common_yaw_acceleration = 30;
  double outpost_yaw_acceleration = 0.01;
  double radius_random_walk = 1e-7;
  double height_random_walk = 1e-7;
  double roll_pitch_random_walk = 0.1;
  double outpost_height_random_walk = 1e-7;
  double armor_match_gate = 200;
  double armor_match_gate_not_all_init = 1000;
  double armor_match_center_weight = 5;
  double armor_match_angle_weight = 10;
  double armor_match_perimeter_weight = 1;
  double uvl_pixel_sigma_ratio = 0.2;
  double uvl_length_sigma_ratio = 0.5;
  double uvl_angle_sigma = 0.1;
  double depth_difference_sigma = 0.1;
};

namespace motion_model
{

namespace index
{
enum StateIndex
{
  CX,
  VCX,
  CY,
  VCY,
  CZ,
  VCZ,
  ROT_Z,
  VYAW,
  LOG_R1,
  P1,
  P2,
  ROT_Y,
  ROT_X,
  STATE_DIM
};
inline constexpr int LOG_R2 = P1;
inline constexpr int H = P2;
inline constexpr int OUTPOST_DZ1 = P1;
inline constexpr int OUTPOST_DZ2 = P2;

enum UVLIndex
{
  UVL_ANGLE,
  UVL_CENTER_X,
  UVL_CENTER_Y,
  UVL_LENGTH,
  UVL_DIM
};
}  // namespace index

inline constexpr int STATE_DIM = index::STATE_DIM;
inline constexpr int UVL_DIM = index::UVL_DIM;
inline constexpr int DIFF_DIM = 1;
inline constexpr double OUTPOST_RADIUS = 0.2765;
inline constexpr double OUTPOST_YAW_RATE = 2.51;
inline constexpr double MIN_RADIUS = 0.05;
inline constexpr double MAX_RADIUS = 1.0;
inline constexpr double NORMAL_ARMOR_PITCH = 15.0 * 3.14159265358979323846 / 180.0;
inline constexpr double OUTPOST_ARMOR_PITCH = -NORMAL_ARMOR_PITCH;

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using Covariance = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
using UVLVector = Eigen::Matrix<double, UVL_DIM, 1>;
using DiffVector = Eigen::Matrix<double, DIFF_DIM, 1>;

inline int armor_count(ArmorName name)
{
  if (name == ArmorName::outpost) return 3;
  if (name == ArmorName::base) return 1;
  return 4;
}

inline bool yaw_only(ArmorName name)
{
  return name == ArmorName::outpost || name == ArmorName::base;
}

template <typename State>
inline auto state_rotation(const State & state, ArmorName name)
{
  using Scalar = typename std::decay_t<State>::Scalar;
  if (yaw_only(name)) {
    return tools::so3_exp(Eigen::Matrix<Scalar, 3, 1>(Scalar(0), Scalar(0), state[index::ROT_Z]));
  }
  return tools::so3_exp(Eigen::Matrix<Scalar, 3, 1>(
    state[index::ROT_X], state[index::ROT_Y], state[index::ROT_Z]));
}

template <typename Delta, typename State>
inline void inject_state(const Delta & delta, State & nominal, ArmorName name)
{
  for (int i = 0; i < STATE_DIM; ++i) {
    if (i != index::ROT_X && i != index::ROT_Y && i != index::ROT_Z) nominal[i] += delta[i];
  }
  if (yaw_only(name)) {
    nominal[index::ROT_Z] = tools::normalize_angle(nominal[index::ROT_Z] + delta[index::ROT_Z]);
    nominal[index::ROT_X] = typename State::Scalar(0);
    nominal[index::ROT_Y] = typename State::Scalar(0);
    return;
  }
  using Scalar = typename std::decay_t<State>::Scalar;
  const Eigen::Matrix<Scalar, 3, 1> delta_rotation(
    delta[index::ROT_X], delta[index::ROT_Y], delta[index::ROT_Z]);
  const auto injected = tools::so3_log(
    (state_rotation(nominal, name) * tools::so3_exp(delta_rotation)).eval());
  nominal[index::ROT_X] = injected.x();
  nominal[index::ROT_Y] = injected.y();
  nominal[index::ROT_Z] = injected.z();
}

template <typename State, typename Delta>
inline void box_minus_state(
  const State & nominal, const State & value, Delta & delta, ArmorName name)
{
  delta = value - nominal;
  if (yaw_only(name)) {
    delta[index::ROT_Z] = tools::normalize_angle(value[index::ROT_Z] - nominal[index::ROT_Z]);
    delta[index::ROT_X] = typename Delta::Scalar(0);
    delta[index::ROT_Y] = typename Delta::Scalar(0);
    return;
  }
  const auto rotation_error = tools::so3_log(
    (state_rotation(nominal, name).transpose() * state_rotation(value, name)).eval());
  delta[index::ROT_X] = rotation_error.x();
  delta[index::ROT_Y] = rotation_error.y();
  delta[index::ROT_Z] = rotation_error.z();
}

template <typename T>
inline void clamp_state(T state[STATE_DIM], ArmorName name)
{
  if (name == ArmorName::outpost) {
    state[index::LOG_R1] = T(std::log(OUTPOST_RADIUS));
    state[index::OUTPOST_DZ1] =
      ceres::fmax(T(-0.3), ceres::fmin(T(0.3), state[index::OUTPOST_DZ1]));
    state[index::OUTPOST_DZ2] =
      ceres::fmax(T(-0.3), ceres::fmin(T(0.3), state[index::OUTPOST_DZ2]));
  } else if (name != ArmorName::base) {
    const T min_log_radius = T(std::log(MIN_RADIUS));
    const T max_log_radius = T(std::log(MAX_RADIUS));
    state[index::LOG_R1] =
      ceres::fmax(min_log_radius, ceres::fmin(max_log_radius, state[index::LOG_R1]));
    state[index::LOG_R2] =
      ceres::fmax(min_log_radius, ceres::fmin(max_log_radius, state[index::LOG_R2]));
    state[index::H] = ceres::fmax(T(-0.5), ceres::fmin(T(0.5), state[index::H]));
  }
  if (yaw_only(name)) {
    state[index::ROT_X] = T(0);
    state[index::ROT_Y] = T(0);
    state[index::ROT_Z] = tools::normalize_angle(state[index::ROT_Z]);
  }
  if (name == ArmorName::base) state[index::VYAW] = T(0);
}

template <typename T>
inline T armor_radius(const T state[STATE_DIM], int id, ArmorName name)
{
  if (name == ArmorName::base) return T(0);
  if (name == ArmorName::outpost) return T(OUTPOST_RADIUS);
  return ceres::exp((id & 1) ? state[index::LOG_R2] : state[index::LOG_R1]);
}

template <typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry> car_pose(
  const T state[STATE_DIM], ArmorName name)
{
  Eigen::Transform<T, 3, Eigen::Isometry> pose =
    Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
  pose.translation() << state[index::CX], state[index::CY], state[index::CZ];
  Eigen::Matrix<T, STATE_DIM, 1> vector;
  for (int i = 0; i < STATE_DIM; ++i) vector[i] = state[i];
  pose.linear() = state_rotation(vector, name);
  return pose;
}

template <typename T>
inline Eigen::Transform<T, 3, Eigen::Isometry> armor_pose(
  const T state[STATE_DIM], int id, ArmorName name)
{
  const int count = armor_count(name);
  const T theta = T(id * 2.0 * 3.14159265358979323846 / count);
  const T radius = armor_radius(state, id, name);
  T height = T(0);
  if (name == ArmorName::outpost) {
    if (id == 1) height = state[index::OUTPOST_DZ1];
    if (id == 2) height = state[index::OUTPOST_DZ2];
  } else if (name != ArmorName::base && (id & 1)) {
    height = state[index::H];
  }

  Eigen::Transform<T, 3, Eigen::Isometry> armor_in_car =
    Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
  armor_in_car.translation() << -ceres::cos(theta) * radius, -ceres::sin(theta) * radius, height;
  const T pitch = T(name == ArmorName::outpost ? OUTPOST_ARMOR_PITCH : NORMAL_ARMOR_PITCH);
  const Eigen::Matrix<T, 3, 1> yaw_vector(T(0), T(0), theta);
  const Eigen::Matrix<T, 3, 1> pitch_vector(T(0), pitch, T(0));
  armor_in_car.linear() = tools::so3_exp(yaw_vector) * tools::so3_exp(pitch_vector);
  return car_pose(state, name) * armor_in_car;
}

struct DirectionVoter
{
  enum State
  {
    collecting,
    clockwise,
    counterclockwise
  } state = collecting;

  void reset(std::chrono::steady_clock::time_point time, double yaw)
  {
    state = collecting;
    start_time = time;
    last_yaw = yaw;
    votes = 0;
  }

  void update(double yaw, std::chrono::steady_clock::time_point time)
  {
    if (time - start_time < std::chrono::seconds(1)) {
      last_yaw = yaw;
      return;
    }
    const double difference = tools::limit_rad(yaw - last_yaw);
    last_yaw = yaw;
    if (std::abs(difference) < 0.05) return;
    votes += difference > 0 ? 1 : -1;
    if (std::abs(votes) > 10) state = votes > 0 ? clockwise : counterclockwise;
  }

  std::chrono::steady_clock::time_point start_time{};
  int votes = 0;
  double last_yaw = 0;
};

struct Predict
{
  double dt = 0;
  ArmorName name = ArmorName::not_armor;
  DirectionVoter::State voter_state = DirectionVoter::collecting;

  template <typename T>
  void operator()(const T previous[STATE_DIM], T predicted[STATE_DIM]) const
  {
    std::copy(previous, previous + STATE_DIM, predicted);
    predicted[index::CX] += previous[index::VCX] * T(dt);
    predicted[index::CY] += previous[index::VCY] * T(dt);
    predicted[index::CZ] += previous[index::VCZ] * T(dt);

    T yaw_rate = previous[index::VYAW];
    if (name == ArmorName::outpost && voter_state != DirectionVoter::collecting) {
      yaw_rate = T(voter_state == DirectionVoter::clockwise ? OUTPOST_YAW_RATE
                                                             : -OUTPOST_YAW_RATE);
      predicted[index::VYAW] = yaw_rate;
    }
    if (name != ArmorName::base) {
      if (yaw_only(name)) {
        predicted[index::ROT_Z] =
          tools::normalize_angle(previous[index::ROT_Z] + yaw_rate * T(dt));
      } else {
        const auto current_rotation = tools::so3_exp(Eigen::Matrix<T, 3, 1>(
          previous[index::ROT_X], previous[index::ROT_Y], previous[index::ROT_Z]));
        const auto delta_rotation = tools::so3_exp(
          Eigen::Matrix<T, 3, 1>(T(0), T(0), yaw_rate * T(dt)));
        const auto rotation_vector = tools::so3_log((current_rotation * delta_rotation).eval());
        predicted[index::ROT_X] = rotation_vector.x();
        predicted[index::ROT_Y] = rotation_vector.y();
        predicted[index::ROT_Z] = rotation_vector.z();
      }
    }
    clamp_state(predicted, name);
  }
};

struct UVLContext
{
  int id = 0;
  bool left = true;
  ArmorName name = ArmorName::not_armor;
  ArmorType type = ArmorType::small;
  CameraContext camera;
};

struct UVLMeasure
{
  UVLContext context;

  template <typename T>
  std::array<Eigen::Matrix<T, 2, 1>, 2> projected_points(const T state[STATE_DIM]) const
  {
    const auto pose_in_world = armor_pose(state, context.id, context.name);
    Eigen::Transform<T, 3, Eigen::Isometry> camera_in_world;
    camera_in_world.matrix() = context.camera.T_camera_world.matrix().template cast<T>();
    const auto pose_in_camera = camera_in_world.inverse() * pose_in_world;
    const auto points = armor_object_points(context.type);
    const int top_index = context.left ? 0 : 1;
    const int bottom_index = context.left ? 3 : 2;
    const auto top_in_camera = pose_in_camera * points[top_index].template cast<T>();
    const auto bottom_in_camera = pose_in_camera * points[bottom_index].template cast<T>();
    return {
      tools::project_point(top_in_camera, context.camera.camera_matrix, context.camera.distortion),
      tools::project_point(bottom_in_camera, context.camera.camera_matrix, context.camera.distortion)};
  }

  template <typename T>
  static void points_to_observation(
    const Eigen::Matrix<T, 2, 1> & top, const Eigen::Matrix<T, 2, 1> & bottom,
    T observation[UVL_DIM])
  {
    const auto difference = top - bottom;
    const auto center = (top + bottom) / T(2);
    observation[index::UVL_ANGLE] = ceres::atan2(difference.x(), difference.y());
    observation[index::UVL_CENTER_X] = center.x();
    observation[index::UVL_CENTER_Y] = center.y();
    observation[index::UVL_LENGTH] = ceres::sqrt(difference.squaredNorm());
  }

  template <typename T>
  void operator()(const T state[STATE_DIM], T observation[UVL_DIM]) const
  {
    const auto points = projected_points(state);
    points_to_observation(points[0], points[1], observation);
  }

  template <typename T>
  static Eigen::Matrix<T, UVL_DIM, 1> residual(
    const Eigen::Matrix<T, UVL_DIM, 1> & predicted,
    const Eigen::Matrix<T, UVL_DIM, 1> & measured)
  {
    Eigen::Matrix<T, UVL_DIM, 1> result = measured - predicted;
    result[index::UVL_ANGLE] = tools::normalize_angle(result[index::UVL_ANGLE]);
    return result;
  }
};

struct DiffMeasure
{
  UVLContext context;

  template <typename T>
  void operator()(const T state[STATE_DIM], T observation[DIFF_DIM]) const
  {
    const auto pose_in_world = armor_pose(state, context.id, context.name);
    Eigen::Transform<T, 3, Eigen::Isometry> camera_in_world;
    camera_in_world.matrix() = context.camera.T_camera_world.matrix().template cast<T>();
    const auto pose_in_camera = camera_in_world.inverse() * pose_in_world;
    const auto points = armor_object_points(context.type);
    const Eigen::Matrix<T, 3, 1> left_center =
      (pose_in_camera * points[0].template cast<T>() +
       pose_in_camera * points[3].template cast<T>()) /
      T(2);
    const Eigen::Matrix<T, 3, 1> right_center =
      (pose_in_camera * points[1].template cast<T>() +
       pose_in_camera * points[2].template cast<T>()) /
      T(2);
    observation[0] = left_center.z() - right_center.z();
  }

  template <typename T>
  static Eigen::Matrix<T, DIFF_DIM, 1> residual(
    const Eigen::Matrix<T, DIFF_DIM, 1> & predicted,
    const Eigen::Matrix<T, DIFF_DIM, 1> & measured)
  {
    return measured - predicted;
  }
};

using RobotESEKF = tools::ErrorStateExtendedKalmanFilter<STATE_DIM, Predict>;

}  // namespace motion_model
}  // namespace auto_aim

#endif  // AUTO_AIM__MOTION_MODEL_HPP
