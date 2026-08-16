#ifndef TOOLS__MATH_TOOLS_HPP
#define TOOLS__MATH_TOOLS_HPP

#include <Eigen/Geometry>
#include <chrono>

namespace tools
{
/** @brief 将弧度角归一化到 (-pi, pi] @param angle 输入角度，单位 rad @return 归一化角度 */
double limit_rad(double angle);

/** @brief 将四元数转换为指定旋转顺序的欧拉角 @param q 四元数 @param axis0 第一旋转轴 @param axis1 第二旋转轴 @param axis2 第三旋转轴 @param extrinsic 是否采用外旋 @return 欧拉角向量，单位 rad */
Eigen::Vector3d eulers(
  Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic = false);

/** @brief 将欧拉角转换为四元数 @param ypr 欧拉角向量 @param axis0 第一旋转轴 @param axis1 第二旋转轴 @param axis2 第三旋转轴 @return 旋转四元数 */
Eigen::Quaterniond toeuler(Eigen::Vector3d ypr, int axis0, int axis1, int axis2);

/** @brief 将旋转矩阵转换为指定旋转顺序的欧拉角 @param R 旋转矩阵 @param axis0 第一旋转轴 @param axis1 第二旋转轴 @param axis2 第三旋转轴 @param extrinsic 是否采用外旋 @return 欧拉角向量，单位 rad */
Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2, bool extrinsic = false);

/** @brief 将 Z-Y-X 欧拉角转换为旋转矩阵 @param ypr 偏航、俯仰、横滚角，单位 rad @return 旋转矩阵 */
Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr);

/** @brief 将直角坐标转换为偏航、俯仰和距离 @param xyz 直角坐标 @return yaw、pitch、distance 向量 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz);

/** @brief 计算 xyz2ypd 对直角坐标的雅可比矩阵 @param xyz 直角坐标 @return 雅可比矩阵 */
Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz);

/** @brief 将偏航、俯仰和距离转换为直角坐标 @param ypd yaw、pitch、distance 向量 @return 直角坐标 */
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd);

/** @brief 计算 ypd2xyz 对球坐标的雅可比矩阵 @param ypd yaw、pitch、distance 向量 @return 雅可比矩阵 */
Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd);

/** @brief 计算时间差 a-b @param a 较新的时间点 @param b 较早的时间点 @return 时间差，单位 s */
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

/** @brief 计算两个二维向量的无符号夹角 @param vec1 向量一 @param vec2 向量二 @return 0 到 pi 的夹角 */
double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2);

/** @brief 计算输入值的平方 @tparam T 数值类型 @param a 输入值 @return a 的平方 */
template <typename T>
T square(T const & a)
{
  return a * a;
};

/** @brief 将输入值限制在给定区间 @param input 输入值 @param min 下限 @param max 上限 @return 限幅后的值 */
double limit_min_max(double input, double min, double max);
}  // namespace tools

#endif  // TOOLS__MATH_TOOLS_HPP
