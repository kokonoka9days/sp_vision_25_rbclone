#include "buff_solver.hpp"

#include <algorithm>
#include <limits>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
double reprojection_error(
  const std::vector<cv::Point3f> & object_points, const std::vector<cv::Point2f> & image_points,
  const cv::Vec3d & rvec, const cv::Vec3d & tvec, const cv::Mat & camera_matrix,
  const cv::Mat & distort_coeffs)
{
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distort_coeffs, projected);
  double error = 0.0;
  for (size_t i = 0; i < projected.size(); ++i) error += cv::norm(projected[i] - image_points[i]);
  return error / static_cast<double>(projected.size());
}

Eigen::Vector3d no_pitch_ypr_from_radial(
  const Eigen::Vector3d & center_in_world, const Eigen::Vector3d & target_in_world,
  const Eigen::Matrix3d & R_target2world)
{
  Eigen::Vector3d radial = target_in_world - center_in_world;
  if (radial.norm() < 1e-6) radial = R_target2world * Eigen::Vector3d(0.0, -1.0, 0.0);
  if (radial.norm() < 1e-6) radial = Eigen::Vector3d::UnitZ();
  radial.normalize();

  const double roll = std::acos(std::clamp(radial.z(), -1.0, 1.0));
  const double sin_roll = std::sin(roll);

  double yaw = 0.0;
  if (std::abs(sin_roll) > 1e-6) {
    yaw = std::atan2(radial.x(), -radial.y());
  } else {
    Eigen::Vector3d normal = R_target2world * Eigen::Vector3d::UnitZ();
    normal.z() = 0.0;
    if (normal.norm() < 1e-6) normal = center_in_world;
    if (normal.norm() < 1e-6) normal = Eigen::Vector3d::UnitX();
    yaw = std::atan2(normal.y(), normal.x());
  }

  return {tools::limit_rad(yaw), 0.0, tools::limit_rad(roll)};
}

cv::Point2f mean_point(const std::vector<cv::Point2f> & points, size_t begin, size_t end)
{
  cv::Point2f sum(0.0f, 0.0f);
  for (size_t i = begin; i < end; ++i) sum += points[i];
  return sum * (1.0f / static_cast<float>(end - begin));
}

double image_angle_around_center(const cv::Point2f & point, const cv::Point2f & center)
{
  return std::atan2(point.y - center.y, point.x - center.x);
}
}  // namespace

cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);
  if (yaml["buff_rune_radius_m"]) RUNE_RADIUS_M = yaml["buff_rune_radius_m"].as<double>();
  if (yaml["buff_small_direction"]) SMALL_BUFF_DIRECTION = yaml["buff_small_direction"].as<int>();

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

std::vector<cv::Point3f> Solver::reproject_object_points() const
{
  const float r = static_cast<float>(RUNE_RADIUS_M);
  return {
    cv::Point3f(0.0f, 0.0f, r + 0.095f), cv::Point3f(0.0f, -0.095f, r),
    cv::Point3f(0.0f, 0.0f, r - 0.095f), cv::Point3f(0.0f, 0.095f, r),
    cv::Point3f(0.0f, 0.030f, r - 0.191f), cv::Point3f(0.0f, -0.030f, r - 0.191f),
    cv::Point3f(0.0f, -0.030f, r - 0.521f), cv::Point3f(0.0f, 0.030f, r - 0.521f),
    cv::Point3f(0.0f, 0.0f, 0.0f)};
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(std::optional<PowerRune> & ps) const
{
  has_pnp_solution_ = false;
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();
  if (p.target().points.size() != 4 || p.target().fan_points.size() != 4) {
    tools::logger()->debug("[Buff_Solver] invalid rune keypoint count");
    return;
  }

  std::vector<cv::Point2f> image_points = p.target().points;
  image_points.insert(image_points.end(), p.target().fan_points.begin(), p.target().fan_points.end());

  std::vector<cv::Vec3d> rvecs;
  std::vector<cv::Vec3d> tvecs;
  cv::Mat reprojection_errors;
  const int solution_count = cv::solvePnPGeneric(
    PNP_OBJECT_POINTS, image_points, camera_matrix_, distort_coeffs_, rvecs, tvecs, false,
    cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), reprojection_errors);
  if (solution_count <= 0 || rvecs.empty()) {
    tools::logger()->debug("[Buff_Solver] solvePnPGeneric failed");
    return;
  }

  int best_index = -1;
  double best_error = std::numeric_limits<double>::max();
  for (int i = 0; i < solution_count; ++i) {
    if (tvecs[i][2] <= 0.0) continue;
    const double error = reprojection_error(
      PNP_OBJECT_POINTS, image_points, rvecs[i], tvecs[i], camera_matrix_, distort_coeffs_);
    if (error < best_error) {
      best_error = error;
      best_index = i;
    }
  }
  if (best_index < 0) {
    tools::logger()->debug("[Buff_Solver] all PnP solutions are behind camera");
    return;
  }

  rvec_ = rvecs[best_index];
  tvec_ = tvecs[best_index];
  cv::solvePnPRefineLM(PNP_OBJECT_POINTS, image_points, camera_matrix_, distort_coeffs_, rvec_, tvec_);
  has_pnp_solution_ = true;

  Eigen::Vector3d t_target2camera;
  cv::cv2eigen(tvec_, t_target2camera);
  cv::Mat rmat;
  cv::Rodrigues(rvec_, rmat);
  Eigen::Matrix3d R_target2camera;
  cv::cv2eigen(rmat, R_target2camera);

  const Eigen::Vector3d center_in_target(0.0, RUNE_RADIUS_M, 0.0);
  const Eigen::Vector3d target_in_camera = t_target2camera;
  const Eigen::Vector3d center_in_camera = R_target2camera * center_in_target + t_target2camera;

  const Eigen::Matrix3d R_target2gimbal = R_camera2gimbal_ * R_target2camera;
  const Eigen::Vector3d target_in_gimbal = R_camera2gimbal_ * target_in_camera + t_camera2gimbal_;
  const Eigen::Vector3d center_in_gimbal = R_camera2gimbal_ * center_in_camera + t_camera2gimbal_;

  const Eigen::Vector3d target_in_world = R_gimbal2world_ * target_in_gimbal;
  const Eigen::Vector3d center_in_world = R_gimbal2world_ * center_in_gimbal;
  const Eigen::Matrix3d R_target2world = R_gimbal2world_ * R_target2gimbal;

  const Eigen::Vector3d ypr_no_pitch =
    no_pitch_ypr_from_radial(center_in_world, target_in_world, R_target2world);

  p.xyz_in_world = center_in_world;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);
  p.blade_xyz_in_world = target_in_world;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);
  p.ypr_in_world = ypr_no_pitch;

  const double roll_probe = 0.02;
  const auto current_points =
    reproject_buff(p.xyz_in_world, p.ypr_in_world[0], p.ypr_in_world[2]);
  const auto positive_roll_points = reproject_buff(
    p.xyz_in_world, p.ypr_in_world[0], tools::limit_rad(p.ypr_in_world[2] + roll_probe));
  if (current_points.size() >= 4 && positive_roll_points.size() >= 4) {
    const double current_angle =
      image_angle_around_center(mean_point(current_points, 0, 4), p.r_center);
    const double positive_roll_angle =
      image_angle_around_center(mean_point(positive_roll_points, 0, 4), p.r_center);
    const double image_delta = tools::limit_rad(positive_roll_angle - current_angle);
    p.positive_roll_image_direction = image_delta >= 0.0 ? 1 : -1;
  } else {
    p.positive_roll_image_direction = 0;
  }
}

cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

std::optional<std::vector<cv::Point2f>> Solver::reproject_pnp_points() const
{
  if (!has_pnp_solution_) return std::nullopt;

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(PNP_OBJECT_POINTS, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    reproject_object_points(), rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
