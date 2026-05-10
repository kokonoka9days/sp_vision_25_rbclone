#include "drone_solver.hpp"

#include <yaml-cpp/yaml.h>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_drone
{
constexpr double DRONE_WIDTH = 0.40;   
constexpr double DRONE_HEIGHT = 0.15;  
constexpr double DRONE_LENGTH = 0.40;  

const std::vector<cv::Point3f> DRONE_POINTS{
  cv::Point3f(-0.025f,    0.01f,   0.0f),
    cv::Point3f(-0.01535f,  0.01f,   0.0146f),
    cv::Point3f( 0.01535f,  0.01f,   0.0146f),
    cv::Point3f( 0.025f,    0.01f,   0.0f),
    cv::Point3f(-0.025f,   -0.01f,   0.0f),
    cv::Point3f(-0.01535f, -0.01f,   0.0146f),
    cv::Point3f( 0.01535f, -0.01f,   0.0146f),
    cv::Point3f( 0.025f,   -0.01f,   0.0f)
};

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

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

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::solve(Drone & drone) const
{
  if (drone.points.size() != 8) {
    tools::logger()->warn("Invalid points size for drone solving! Expected 8 points.");
    return;
  }

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    DRONE_POINTS, drone.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_EPNP);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  drone.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  drone.xyz_in_world = R_gimbal2world_ * drone.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  
  drone.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  drone.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  drone.ypd_in_world = tools::xyz2ypd(drone.xyz_in_world);
}

std::vector<cv::Point2f> Solver::reproject_drone(
  const Eigen::Vector3d & xyz_in_world, const Eigen::Vector3d & ypr_in_world) const
{
  Eigen::Matrix3d R_drone2world = tools::rotation_matrix(ypr_in_world);

  const Eigen::Vector3d & t_drone2world = xyz_in_world;
  Eigen::Matrix3d R_drone2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_drone2world;
  Eigen::Vector3d t_drone2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_drone2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_drone2camera_cv;
  cv::eigen2cv(R_drone2camera, R_drone2camera_cv);
  cv::Rodrigues(R_drone2camera_cv, rvec);
  cv::Vec3d tvec(t_drone2camera[0], t_drone2camera[1], t_drone2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(DRONE_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  
  return image_points;
}

std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}

}  // namespace auto_drone