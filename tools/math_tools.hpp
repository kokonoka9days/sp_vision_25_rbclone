#ifndef TOOLS__MATH_TOOLS_HPP
#define TOOLS__MATH_TOOLS_HPP

#include <Eigen/Geometry>
#include <ceres/jet.h>
#include <chrono>
#include <cmath>
#include <limits>

namespace tools
{
template <typename T>
inline Eigen::Matrix<T, 3, 3> so3_hat(const Eigen::Matrix<T, 3, 1> & w)
{
  Eigen::Matrix<T, 3, 3> W;
  W << T(0), -w.z(), w.y(), w.z(), T(0), -w.x(), -w.y(), w.x(), T(0);
  return W;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> so3_exp(const Eigen::Matrix<T, 3, 1> & phi)
{
  const T theta2 = phi.squaredNorm();
  const T theta = ceres::sqrt(theta2);
  const auto W = so3_hat(phi);
  const auto W2 = W * W;

  T A;
  T B;
  if (theta2 < T(1e-12)) {
    const T theta4 = theta2 * theta2;
    A = T(1) - theta2 / T(6) + theta4 / T(120);
    B = T(0.5) - theta2 / T(24) + theta4 / T(720);
  } else {
    A = ceres::sin(theta) / theta;
    B = (T(1) - ceres::cos(theta)) / theta2;
  }
  return Eigen::Matrix<T, 3, 3>::Identity() + A * W + B * W2;
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> so3_log(const Eigen::Matrix<T, 3, 3> & R)
{
  Eigen::Quaternion<T> q(R);
  if (q.w() < T(0)) q.coeffs() = -q.coeffs();
  const Eigen::Matrix<T, 3, 1> v = q.vec();
  const T v_norm = ceres::sqrt(v.squaredNorm());
  if (v_norm < T(1e-12)) return T(2) * v;
  const T angle = T(2) * ceres::atan2(v_norm, q.w());
  return (angle / v_norm) * v;
}

template <typename T>
inline T normalize_angle(T angle)
{
  const T pi = T(3.14159265358979323846);
  const T two_pi = T(2) * pi;
  return angle - two_pi * ceres::floor((angle + pi) / two_pi);
}

template <typename T>
inline double scalar_value(const T & value)
{
  return static_cast<double>(value);
}

template <typename T, int N>
inline double scalar_value(const ceres::Jet<T, N> & value)
{
  return static_cast<double>(value.a);
}

template <typename T>
inline Eigen::Matrix<T, 2, 1> project_point(
  const Eigen::Matrix<T, 3, 1> & point_in_camera, const Eigen::Matrix3d & camera_matrix,
  const Eigen::Matrix<double, 5, 1> & distortion)
{
  const double depth = scalar_value(point_in_camera.z());
  if (!std::isfinite(depth) || depth <= 1e-9) {
    const T invalid = T(std::numeric_limits<double>::quiet_NaN());
    return {invalid, invalid};
  }
  const T x = point_in_camera.x() / point_in_camera.z();
  const T y = point_in_camera.y() / point_in_camera.z();
  const T r2 = x * x + y * y;
  const T radial =
    T(1) + T(distortion[0]) * r2 + T(distortion[1]) * r2 * r2 +
    T(distortion[4]) * r2 * r2 * r2;
  const T xd = x * radial + T(2 * distortion[2]) * x * y + T(distortion[3]) * (r2 + T(2) * x * x);
  const T yd = y * radial + T(distortion[2]) * (r2 + T(2) * y * y) + T(2 * distortion[3]) * x * y;
  return {
    T(camera_matrix(0, 0)) * xd + T(camera_matrix(0, 2)),
    T(camera_matrix(1, 1)) * yd + T(camera_matrix(1, 2))};
}

// 将弧度值限制在(-pi, pi]
double limit_rad(double angle);

// 四元数转欧拉角
// x = 0, y = 1, z = 2
// e.g. 先绕z轴旋转，再绕y轴旋转，最后绕x轴旋转：axis0=2, axis1=1, axis2=0
// 参考：https://github.com/evbernardes/quaternion_to_euler
Eigen::Vector3d eulers(
  Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic = false);

//欧拉角转四元数
Eigen::Quaterniond toeuler(Eigen::Vector3d ypr, int axis0, int axis1, int axis2);

// 旋转矩阵转欧拉角
// x = 0, y = 1, z = 2
// e.g. 先绕z轴旋转，再绕y轴旋转，最后绕x轴旋转：axis0=2, axis1=1, axis2=0
Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2, bool extrinsic = false);

// 欧拉角转旋转矩阵
// zyx:先绕z轴旋转，再绕y轴旋转，最后绕x轴旋转
Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr);

// 直角坐标系转球坐标系
// ypd为yaw、pitch、distance的缩写
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz);

// 直角坐标系转球坐标系转换函数对xyz的雅可比矩阵
Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz);

// 球坐标系转直角坐标系
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd);

// 球坐标系转直角坐标系转换函数对xyz的雅可比矩阵
Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd);

// 计算时间差a - b，单位：s
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

// 向量夹角 总是返回 0 ~ pi 来自SJTU
double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2);

// 返回输入值的平方
template <typename T>
T square(T const & a)
{
  return a * a;
};

double limit_min_max(double input, double min, double max);
}  // namespace tools

#endif  // TOOLS__MATH_TOOLS_HPP
