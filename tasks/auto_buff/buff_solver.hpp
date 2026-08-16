#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "buff_type.hpp"
#include "buff_config.hpp"
#include "tools/math_tools.hpp"
namespace auto_buff
{
// 旋转角度
const double THETA = 2.0 * CV_PI / 5.0;  // 2/5π

class Solver
{
public:
  /** @brief 从配置文件初始化能量机关位姿求解器 @param config_path YAML 配置路径 */
  explicit Solver(const std::string & config_path);
  /** @brief 使用已解析配置初始化位姿求解器 @param config_path YAML 配置路径 @param config 能量机关配置 */
  Solver(const std::string & config_path, BuffConfig config);

  /** @brief 获取云台到世界坐标系旋转 @return 旋转矩阵 */
  Eigen::Matrix3d R_gimbal2world() const;

  /** @brief 由 IMU 四元数更新云台到世界坐标系旋转 @param q 姿态四元数 */
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  /** @brief 求解单个能量机关观测 @param observation 图像观测 @return 成功时返回三维能量机关 */
  std::optional<PowerRune> solve(const BuffObservation & observation) const;

  /** @brief 求解可选观测 @param observation 可选图像观测 @return 成功时返回三维能量机关 */
  std::optional<PowerRune> solve(
    const std::optional<BuffObservation> & observation) const;

  /** @brief 批量求解观测 @param observations 图像观测列表 @return 成功求解的三维结果 */
  std::vector<PowerRune> solve_all(const std::vector<BuffObservation> & observations) const;

  /** @brief 将能量机关坐标点投影到像素平面 @param x 能量机关坐标点 @return 像素坐标 */
  cv::Point2f point_buff2pixel(cv::Point3f x);

  /** @brief 重投影最近一次 PnP 使用的对象点 @return 有有效 PnP 解时返回像素点列表 */
  std::optional<std::vector<cv::Point2f>> reproject_pnp_points() const;

  /** @brief 将指定姿态的能量机关模型重投影到图像 @param xyz_in_world 中心世界坐标 @param R_buff2world 能量机关到世界旋转 @return 像素点列表 */
  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d & R_buff2world) const;

private:
  const BuffConfig config_;
  mutable std::recursive_mutex mutex_;
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  mutable cv::Vec3d rvec_, tvec_;
  mutable bool has_pnp_solution_ = false;
  struct PoseCache
  {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
  };
  mutable std::unordered_map<int, PoseCache> pose_cache_;

  double full_reprojection_gate_px_ = 6.0;
  double target_center_reprojection_gate_px_ = 6.0;
  double fan_center_reprojection_gate_px_ = 8.0;
  double partial_four_center_gate_px_ = 8.0;
  double partial_four_angle_gate_rad_ = 15.0 / 57.3;

  const std::vector<cv::Point3f> PNP_OBJECT_POINTS = {
    cv::Point3f(0.0f, -0.095f, 0.0f), cv::Point3f(0.095f, 0.0f, 0.0f),
    cv::Point3f(0.0f, 0.095f, 0.0f), cv::Point3f(-0.095f, 0.0f, 0.0f),
    cv::Point3f(-0.030f, 0.191f, 0.0f), cv::Point3f(0.030f, 0.191f, 0.0f),
    cv::Point3f(0.030f, 0.521f, 0.0f), cv::Point3f(-0.030f, 0.521f, 0.0f)};

  /** @brief 获取用于重投影的三维模型点 @return 模型点列表 */
  std::vector<cv::Point3f> reproject_object_points() const;

  /** @brief 生成绕 X 轴的旋转矩阵 @param angle 旋转角，单位 rad @return 旋转矩阵 */
  cv::Matx33f rotation_matrix(double angle) const;

  /** @brief 生成各扇叶槽位的旋转模型点 @param object_points 输出各槽位模型点 */
  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);
};
}  // namespace auto_buff
#endif  // AUTO_AIM__SOLVER_HPP
