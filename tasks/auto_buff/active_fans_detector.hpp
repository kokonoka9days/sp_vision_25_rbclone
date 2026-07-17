#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace auto_buff::big
{
class ActiveFanDetector
{
public:
  using Roi = cv::Rect2f;
  using Contour = std::vector<cv::Point>;

  enum class RoiContourSelection
  {
    LARGEST_AREA,
    NEAREST_CENTER
  };

  struct IndexedContour
  {
    int id = -1;
    Contour points;
    double area = 0.0;
    cv::Point2f center{};
  };

  struct DetectionResult
  {
    cv::Mat binary;
    Roi big_roi;

    // findContours 产生的全部轮廓，id 是轮廓的原始序号。
    std::vector<IndexedContour> detected_contours;
    std::vector<int> outside_big_roi_ids;

    // 与输入 rois 一一对应；该 ROI 没有可剔除轮廓时为 -1。
    std::vector<int> roi_excluded_ids;
    std::vector<IndexedContour> remaining_contours;
  };

  /**
   * @brief 从 YAML 配置文件创建激活扇叶轮廓检测器。
   * @param config YAML 配置文件路径。
   */
  explicit ActiveFanDetector(const std::string & config);

  /**
   * @brief 提取、编号并按照中心框和外部识别框过滤图像轮廓。
   * @param bgr_img 输入的 CV_8UC3 BGR 图像。
   * @param r_roi 中心 R 的识别框，用于生成放大后的 big ROI。
   * @param rois 来源于输入图像的外部深度识别框。
   * @return 二值图、轮廓编号、剔除编号及最终保留轮廓。
   */
  DetectionResult detect(
    const cv::Mat & bgr_img, const Roi & r_roi, const std::vector<Roi> & rois) const;

private:
  enum class PixChannel : int
  {
    BLUE = 0,
    RED = 2
  };

  float roi_scale_ = 2.0f;
  PixChannel color_ = PixChannel::BLUE;
  std::uint8_t color_thresh_ = 50;
  RoiContourSelection roi_contour_selection_ = RoiContourSelection::LARGEST_AREA;

  /** @brief 根据目标颜色通道差生成二值图。 */
  cv::Mat binarize(const cv::Mat & bgr_img) const;
};
}  // namespace auto_buff::big
