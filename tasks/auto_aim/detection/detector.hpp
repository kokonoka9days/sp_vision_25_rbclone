#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor.hpp"
#include "classifier.hpp"

namespace auto_aim
{

class Detector
{
public:
  /** @brief 初始化传统视觉装甲板检测器 @param config_path YAML 配置文件路径 @param debug 是否保存或显示调试结果 */
  Detector(const std::string & config_path, bool debug = true);

  /** @brief 检测图像中的全部装甲板 @param bgr_img BGR 输入图像 @param frame_count 调试帧编号 @return 检测到的装甲板列表 */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

  /** @brief 检查指定装甲板在图像中是否仍可检测 @param armor 输入及输出装甲板 @param bgr_img BGR 输入图像 @return 检测成功时返回 true */
  bool detect(Armor & armor, const cv::Mat & bgr_img);

  friend class YOLOV8;

private:
  Classifier classifier_;

  double threshold_;
  double max_angle_error_;
  double min_lightbar_ratio_, max_lightbar_ratio_;
  double min_lightbar_length_;
  double min_armor_ratio_, max_armor_ratio_;
  double max_side_ratio_;
  double min_confidence_;
  double max_rectangular_error_;

  bool debug_;
  std::string save_path_;

  /** @brief 使用 PCA 修正灯条端点 @param lightbar 待修正灯条 @param gray_img 灰度图像 */
  void lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const;

  /** @brief 检查灯条几何约束 @param lightbar 灯条 @return 通过约束时返回 true */
  bool check_geometry(const Lightbar & lightbar) const;
  /** @brief 检查装甲板几何约束 @param armor 装甲板 @return 通过约束时返回 true */
  bool check_geometry(const Armor & armor) const;
  /** @brief 检查装甲板类别是否有效 @param armor 装甲板 @return 类别有效时返回 true */
  bool check_name(const Armor & armor) const;
  /** @brief 检查装甲板尺寸类型是否有效 @param armor 装甲板 @return 类型有效时返回 true */
  bool check_type(const Armor & armor) const;

  /** @brief 根据轮廓内颜色判断灯条颜色 @param bgr_img BGR 图像 @param contour 灯条轮廓 @return 灯条颜色 */
  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;
  /** @brief 提取用于数字分类的装甲板图案 @param bgr_img BGR 图像 @param armor 装甲板 @return 透视校正后的图案 */
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;
  /** @brief 根据几何比例判断装甲板尺寸 @param armor 装甲板 @return 大或小装甲板类型 */
  ArmorType get_type(const Armor & armor);
  /** @brief 将像素中心归一化 @param bgr_img 输入图像 @param center 像素中心 @return 归一化坐标 */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /** @brief 保存装甲板图案用于数据采集 @param armor 装甲板 */
  void save(const Armor & armor) const;
  /** @brief 绘制并显示检测调试结果 @param binary_img 二值图 @param bgr_img 原始 BGR 图像 @param lightbars 灯条列表 @param armors 装甲板列表 @param frame_count 帧编号 */
  void show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
    const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__DETECTOR_HPP
