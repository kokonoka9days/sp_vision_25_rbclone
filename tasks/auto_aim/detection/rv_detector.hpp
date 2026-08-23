#ifndef RV_AIM__DETECTOR_HPP_
#define RV_AIM__DETECTOR_HPP_

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

// 引用 auto_aim 的基础数据结构
#include "../model/armor.hpp"
// 引用 rv_aim 的数字识别器
#include "rv_number.hpp"

namespace rv_aim
{

class Detector
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径
   * @param debug 是否开启调试显示
   */
  Detector(const std::string & config_path, bool debug = true);

  /**
   * @brief 每一帧的主检测函数
   * @param bgr_img 输入图像
   * @param frame_count 帧计数（用于Debug显示）
   * @return 识别到的装甲板列表
   */
  std::list<auto_aim::Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

private:
  // 使用 rv_aim 的分类器
std::shared_ptr<NumberClassifier>  classifier_;

  // 几何筛选参数 (与 auto_aim 保持一致)
  double threshold_;
  double max_angle_error_;
  double min_lightbar_ratio_, max_lightbar_ratio_;
  double min_lightbar_length_;
  double min_armor_ratio_, max_armor_ratio_;
  double max_side_ratio_;
  double min_confidence_;
  double max_rectangular_error_;

  bool debug_;
  
  /** @brief 检查灯条几何约束 @param lightbar 灯条 @return 通过约束时返回 true */
  bool check_geometry(const auto_aim::Lightbar & lightbar) const;
  /** @brief 检查装甲板几何约束 @param armor 装甲板 @return 通过约束时返回 true */
  bool check_geometry(const auto_aim::Armor & armor) const;
  
  /** @brief 根据轮廓内颜色判断灯条颜色 @param bgr_img BGR 图像 @param contour 灯条轮廓 @return 灯条颜色 */
  auto_aim::Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;
  /** @brief 根据几何比例判断装甲板尺寸 @param armor 装甲板 @return 大或小装甲板类型 */
  auto_aim::ArmorType get_type(const auto_aim::Armor & armor); // 仅基于几何判断大小装甲

  /** @brief 绘制检测调试结果 @param binary_img 二值图 @param bgr_img 原始图像 @param lightbars 灯条列表 @param armors 装甲板列表 @param frame_count 帧编号 */
  void show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<auto_aim::Lightbar> & lightbars,
    const std::vector<auto_aim::Armor> & armors, int frame_count) const;
};

}  // namespace rv_aim

#endif  // RV_AIM__DETECTOR_HPP_
