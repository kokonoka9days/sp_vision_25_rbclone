#ifndef TOOLS__REPROJECTION_HPP
#define TOOLS__REPROJECTION_HPP

#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <opencv2/core.hpp>
#include <vector>

namespace tools
{
using ArmorReprojector =
  std::function<std::vector<cv::Point2f>(const Eigen::Vector3d &, double)>;

/**
 * @brief 将目标装甲板和瞄准点重投影到图像
 * @param image 输出图像
 * @param ekf_x 目标滤波状态
 * @param armor_xyza_list 各装甲板的世界坐标与偏航角
 * @param aim_xyza 可选瞄准点世界坐标与偏航角
 * @param reproject_armor 装甲板重投影函数
 * @param armor_color 装甲板轮廓颜色
 * @param aim_color 瞄准点颜色
 */
void draw_reprojection(
  cv::Mat & image, const Eigen::VectorXd & ekf_x,
  const std::vector<Eigen::Vector4d> & armor_xyza_list,
  const std::optional<Eigen::Vector4d> & aim_xyza, const ArmorReprojector & reproject_armor,
  const cv::Scalar & armor_color = {0, 255, 0},
  const cv::Scalar & aim_color = {0, 0, 255});

/** @brief 使用求解器和目标对象绘制重投影 @param image 输出图像 @param solver 位姿求解器 @param target 跟踪目标 @param aim_xyza 可选瞄准点 @param armor_color 装甲板轮廓颜色 @param aim_color 瞄准点颜色 */
template<typename Solver, typename Target>
void draw_reprojection(
  cv::Mat & image, const Solver & solver, const Target & target,
  const std::optional<Eigen::Vector4d> & aim_xyza = std::nullopt,
  const cv::Scalar & armor_color = {0, 255, 0},
  const cv::Scalar & aim_color = {0, 0, 255})
{
  auto reproject_armor = [&](const Eigen::Vector3d & xyz, double yaw) {
    return solver.reproject_armor(xyz, yaw, target.armor_type, target.name);
  };
  draw_reprojection(
    image, target.ekf_x(), target.armor_xyza_list(), aim_xyza, reproject_armor, armor_color,
    aim_color);
}

}  // namespace tools

#endif  // TOOLS__REPROJECTION_HPP
