#include "rv_from_fyt.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr double kTowerArmorLargeHeightJump = 0.16;
constexpr double kTowerArmorSmallHeightJump = 0.05;
constexpr double kChiSquareThreshold = 9.488;
constexpr double kHysteresisMargin = 5.0;
}  // namespace

RVfromFYT::RVfromFYT(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0, int armor_num,
  ArmorName armor_name)
: tools::ExtendedKalmanFilter(x0, P0, state_add),
  armor_num_(armor_num),
  armor_name_(armor_name)
{
  if (x0.size() != kStateDimension || P0.rows() != kStateDimension ||
      P0.cols() != kStateDimension) {
    throw std::invalid_argument("RVfromFYT requires an 11-dimensional state");
  }
  if (armor_num_ <= 0) {
    throw std::invalid_argument("RVfromFYT requires at least one armor");
  }
}

void RVfromFYT::predict_model(
  double dt, const Eigen::Vector3d & acceleration, double position_process_noise,
  double yaw_process_noise)
{
  // One constant-velocity block for [position, velocity].
  Eigen::Matrix2d constant_velocity_transition;
  constant_velocity_transition << 1.0, dt,
                                  0.0, 1.0;

  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kStateDimension, kStateDimension);
  F.block<2, 2>(0, 0) = constant_velocity_transition;  // x, vx
  F.block<2, 2>(2, 2) = constant_velocity_transition;  // y, vy
  F.block<2, 2>(4, 4) = constant_velocity_transition;  // z, vz
  F.block<2, 2>(6, 6) = constant_velocity_transition;  // yaw, vyaw

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  // One white-noise acceleration block for the process covariance.
  auto process_noise_block = [&](double variance) {
    Eigen::Matrix2d block;
    block << dt4 / 4.0 * variance, dt3 / 2.0 * variance,
             dt3 / 2.0 * variance, dt2 * variance;
    return block;
  };

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateDimension, kStateDimension);
  Q.block<2, 2>(0, 0) = process_noise_block(position_process_noise);  // x, vx
  Q.block<2, 2>(2, 2) = process_noise_block(position_process_noise);  // y, vy
  Q.block<2, 2>(4, 4) = process_noise_block(position_process_noise);  // z, vz
  Q.block<2, 2>(6, 6) = process_noise_block(yaw_process_noise);       // yaw, vyaw

  // State: [x, vx, y, vy, z, vz, yaw, vyaw, radius, radius_offset, height_offset].
  // Input: [ax, ay, az].
  const double half_dt2 = 0.5 * dt2;
  Eigen::Matrix<double, kStateDimension, 3> B;
  B << half_dt2, 0.0,      0.0,       // x
       dt,       0.0,      0.0,       // vx
       0.0,      half_dt2, 0.0,       // y
       0.0,      dt,       0.0,       // vy
       0.0,      0.0,      half_dt2,  // z
       0.0,      0.0,      dt,        // vz
       0.0,      0.0,      0.0,       // yaw
       0.0,      0.0,      0.0,       // vyaw
       0.0,      0.0,      0.0,       // radius
       0.0,      0.0,      0.0,       // radius offset
       0.0,      0.0,      0.0;       // height offset

  auto f = [&](const Eigen::VectorXd & state) {
    Eigen::VectorXd prediction = F * state + B * acceleration;
    prediction[6] = tools::limit_rad(prediction[6]);
    return prediction;
  };

  tools::ExtendedKalmanFilter::predict(F, Q, f);
}

void RVfromFYT::prepare_measurement(
  const Armor & armor, bool cam_is_short, int update_count)
{
  if (last_cam_is_short_ != cam_is_short) {
    camera_switch_time_ = std::chrono::steady_clock::now();
    last_cam_is_short_ = cam_is_short;
  }

  const double center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const double delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  double azimuth_variance = 4e-3;
  const double pitch_variance = 4e-3;
  double distance_variance = std::log(std::abs(delta_angle) + 1.0) + 1.0;
  double angle_variance = std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 + 9e-2;

  const double time_since_camera_switch = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - camera_switch_time_)
                                            .count();
  if (time_since_camera_switch < 0.7 && update_count > 50) {
    azimuth_variance = 4e4;
    distance_variance *= 300.0;
    angle_variance *= 300.0;
  }

  // Measurement: [yaw, pitch, distance, armor_yaw].
  R_ << azimuth_variance, 0.0,            0.0,               0.0,             // yaw
        0.0,              pitch_variance, 0.0,               0.0,             // pitch
        0.0,              0.0,            distance_variance, 0.0,             // distance
        0.0,              0.0,            0.0,               angle_variance;  // armor yaw

  z_ << armor.ypd_in_world[0],
        armor.ypd_in_world[1],
        armor.ypd_in_world[2],
        armor.ypr_in_world[0];
  measurement_ready_ = true;
}

int RVfromFYT::select_armor_id(int last_id) const
{
  if (!measurement_ready_) {
    throw std::logic_error("RVfromFYT measurement must be prepared before armor selection");
  }

  int best_id = 0;
  double min_distance = std::numeric_limits<double>::infinity();
  std::vector<double> distances(armor_num_, std::numeric_limits<double>::infinity());

  for (int armor_id = 0; armor_id < armor_num_; ++armor_id) {
    const Eigen::VectorXd residual = observation_subtract(z_, h(x, armor_id));
    const Eigen::Matrix<double, 4, kStateDimension> H = h_jacobian(x, armor_id);
    const Eigen::Matrix4d S = H * P * H.transpose() + R_;
    const Eigen::LDLT<Eigen::Matrix4d> decomposition(S);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) continue;

    const double distance = residual.dot(decomposition.solve(residual));
    distances[armor_id] = distance;
    if (distance < min_distance) {
      min_distance = distance;
      best_id = armor_id;
    }
  }

  if (
    last_id >= 0 && last_id < armor_num_ && distances[last_id] < kChiSquareThreshold &&
    min_distance > distances[last_id] - kHysteresisMargin) {
    return last_id;
  }
  return best_id;
}

Eigen::VectorXd RVfromFYT::correct(int armor_id)
{
  if (!measurement_ready_) {
    throw std::logic_error("RVfromFYT measurement must be prepared before correction");
  }

  const Eigen::Matrix<double, 4, kStateDimension> H = h_jacobian(x, armor_id);
  auto observation_model = [&](const Eigen::VectorXd & state) { return h(state, armor_id); };
  measurement_ready_ = false;
  return tools::ExtendedKalmanFilter::update(
    z_, H, R_, observation_model, observation_subtract);
}

void RVfromFYT::set_tower_armor_heights(const std::pair<bool, double> (&heights)[3])
{
  for (std::size_t i = 0; i < tower_armor_heights_.size(); ++i) {
    tower_armor_heights_[i] = heights[i].second;
  }
}

Eigen::Vector3d RVfromFYT::h_armor_xyz(
  const Eigen::VectorXd & state, int armor_id) const
{
  const double angle =
    tools::limit_rad(state[6] + armor_id * 2.0 * CV_PI / armor_num_);
  const bool use_alternate_radius = armor_num_ == 4 && (armor_id == 1 || armor_id == 3);
  const double radius = use_alternate_radius ? state[8] + state[9] : state[8];

  const double armor_x = state[0] - radius * std::cos(angle);
  const double armor_y = state[2] - radius * std::sin(angle);
  const double armor_z = armor_name_ == ArmorName::outpost
                           ? state[4] + state[10] * tower_height_multiplier(armor_id)
                           : state[4] + (use_alternate_radius ? state[10] : 0.0);
  return {armor_x, armor_y, armor_z};
}

std::vector<Eigen::Vector4d> RVfromFYT::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> armors;
  armors.reserve(armor_num_);
  for (int armor_id = 0; armor_id < armor_num_; ++armor_id) {
    const Eigen::Vector3d xyz = h_armor_xyz(x, armor_id);
    const double angle = tools::limit_rad(x[6] + armor_id * 2.0 * CV_PI / armor_num_);
    armors.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return armors;
}

Eigen::Vector4d RVfromFYT::h(const Eigen::VectorXd & state, int armor_id) const
{
  const Eigen::Vector3d ypd = tools::xyz2ypd(h_armor_xyz(state, armor_id));
  const double angle =
    tools::limit_rad(state[6] + armor_id * 2.0 * CV_PI / armor_num_);

  Eigen::Vector4d observation;
  observation << ypd[0], ypd[1], ypd[2], angle;
  return observation;
}

Eigen::Matrix<double, 4, RVfromFYT::kStateDimension> RVfromFYT::h_jacobian(
  const Eigen::VectorXd & state, int armor_id) const
{
  const double angle =
    tools::limit_rad(state[6] + armor_id * 2.0 * CV_PI / armor_num_);
  const bool use_alternate_radius = armor_num_ == 4 && (armor_id == 1 || armor_id == 3);
  const double radius = use_alternate_radius ? state[8] + state[9] : state[8];

  const double dx_dyaw = radius * std::sin(angle);
  const double dy_dyaw = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);
  const double dx_dr_offset = use_alternate_radius ? dx_dr : 0.0;
  const double dy_dr_offset = use_alternate_radius ? dy_dr : 0.0;
  const double dz_dh = armor_name_ == ArmorName::outpost
                         ? tower_height_multiplier(armor_id)
                         : (use_alternate_radius ? 1.0 : 0.0);

  // Jacobian from the filter state to armor [x, y, z, yaw].
  Eigen::Matrix<double, 4, kStateDimension> H_xyza;
  H_xyza << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, dx_dyaw, 0.0, dx_dr, dx_dr_offset, 0.0,   // x
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dy_dyaw, 0.0, dy_dr, dy_dr_offset, 0.0,   // y
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,     0.0, 0.0,   0.0,          dz_dh, // z
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,     0.0, 0.0,   0.0,          0.0;   // yaw

  const Eigen::Matrix3d H_ypd = tools::xyz2ypd_jacobian(h_armor_xyz(state, armor_id));
  // Jacobian from armor [x, y, z, yaw] to [yaw, pitch, distance, armor_yaw].
  Eigen::Matrix4d H_ypda;
  H_ypda << H_ypd(0, 0), H_ypd(0, 1), H_ypd(0, 2), 0.0,  // yaw
            H_ypd(1, 0), H_ypd(1, 1), H_ypd(1, 2), 0.0,  // pitch
            H_ypd(2, 0), H_ypd(2, 1), H_ypd(2, 2), 0.0,  // distance
            0.0,         0.0,         0.0,         1.0;  // armor yaw
  return H_ypda * H_xyza;
}

double RVfromFYT::tower_height_multiplier(int armor_id) const
{
  const double height_delta = tower_armor_heights_[armor_id] - tower_armor_heights_[0];
  const double direction = height_delta > 0.0 ? 1.0 : -1.0;

  int step_count = 0;
  if (std::abs(height_delta) > kTowerArmorLargeHeightJump) {
    step_count = 2;
  } else if (std::abs(height_delta) > kTowerArmorSmallHeightJump) {
    step_count = 1;
  }
  return direction * step_count;
}

Eigen::VectorXd RVfromFYT::state_add(
  const Eigen::VectorXd & state, const Eigen::VectorXd & delta)
{
  Eigen::VectorXd result = state + delta;
  result[6] = tools::limit_rad(result[6]);
  return result;
}

Eigen::VectorXd RVfromFYT::observation_subtract(
  const Eigen::VectorXd & observation, const Eigen::VectorXd & prediction)
{
  Eigen::VectorXd residual = observation - prediction;
  residual[0] = tools::limit_rad(residual[0]);
  residual[1] = tools::limit_rad(residual[1]);
  residual[3] = tools::limit_rad(residual[3]);
  return residual;
}

}  // namespace auto_aim
