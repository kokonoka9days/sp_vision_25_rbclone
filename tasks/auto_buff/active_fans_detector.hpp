#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo11_buff.hpp"

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
    int rune_center_contour_id = -1;
    double rune_center_contour_area = 0.0;
    std::vector<int> smaller_than_rune_center_ids;

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
   * @brief 提取、编号，并按中心框、中心轮廓面积和外部识别框过滤轮廓。
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


using ActiveFanDetector = auto_buff::big::ActiveFanDetector;

/**
 * @brief 将浮点检测框转换为整型矩形，并裁剪到图像边界内。
 * @param rect YOLO 输出的浮点检测框。
 * @param image_size 输入图像尺寸。
 * @return 可安全用于 OpenCV 绘制的整型矩形。
 */
inline cv::Rect integer_rect(const cv::Rect2f & rect, const cv::Size & image_size)
{
  const int left = static_cast<int>(std::floor(rect.x));
  const int top = static_cast<int>(std::floor(rect.y));
  const int right = static_cast<int>(std::ceil(rect.x + rect.width));
  const int bottom = static_cast<int>(std::ceil(rect.y + rect.height));
  return cv::Rect(cv::Point(left, top), cv::Point(right, bottom)) &
         cv::Rect(0, 0, image_size.width, image_size.height);
}

/**
 * @brief 根据类别编号查询 YOLO 类别名称。
 * @param label YOLO 输出的类别编号。
 * @return 类别名称；类别无效时返回 unknown。
 */
inline std::string label_name(int label)
{
  if (label < 0 || label >= static_cast<int>(auto_buff::class_names.size())) return "unknown";
  return auto_buff::class_names[label];
}

/**
 * @brief 绘制深度模型识别结果及每个 ROI 对应的轮廓剔除编号。
 * @param image 输出可视化图像。
 * @param objects YOLO 输出的全部识别对象。
 * @param r_object_index 用于生成 big ROI 的 rune_center 对象下标。
 * @param roi_object_indexes ROI 下标到 YOLO 对象下标的映射。
 * @param roi_excluded_ids 每个 ROI 剔除的轮廓编号，未命中时为 -1。
 */
inline void draw_deep_rois(
  cv::Mat & image, const std::vector<auto_buff::YOLO11_BUFF::Object> & objects,
  int r_object_index, const std::vector<int> & roi_object_indexes,
  const std::vector<int> & roi_excluded_ids)
{
  std::vector<int> object_to_roi(objects.size(), -1);
  for (std::size_t i = 0; i < roi_object_indexes.size(); ++i) {
    object_to_roi[static_cast<std::size_t>(roi_object_indexes[i])] = static_cast<int>(i);
  }

/**
 * @brief 根据 YOLO 类别编号返回对应的 BGR 可视化颜色。
 * @param label YOLO 输出的类别编号。
 * @return 类别有效时返回预设颜色，否则返回白色。
 */

  auto color_for_label = [](int label) -> cv::Scalar {
    static const std::vector<cv::Scalar> colors = {
      {0, 220, 255}, {255, 160, 0}, {80, 255, 80}};
    if (label < 0 || label >= static_cast<int>(colors.size())) return {255, 255, 255};
    return colors[label];
  };


  for (std::size_t i = 0; i < objects.size(); ++i) {
    const auto & object = objects[i];
    const cv::Rect box = integer_rect(object.rect, image.size());
    if (box.empty()) continue;

    const cv::Scalar color = color_for_label(object.label);
    // cv::rectangle(image, box, color, i == static_cast<std::size_t>(r_object_index) ? 3 : 2);

    std::string text = label_name(object.label) + cv::format(" %.2f", object.prob);
    const int roi_index = object_to_roi[i];
    if (roi_index >= 0) {
      text += " roi[" + std::to_string(roi_index) + "] -> contour[" +
              std::to_string(roi_excluded_ids[static_cast<std::size_t>(roi_index)]) + "]";
    }
    cv::putText(
      image, text, cv::Point(box.x, std::max(14, box.y - 4)), cv::FONT_HERSHEY_SIMPLEX,
      0.45, color, 1, cv::LINE_AA);
  }
}

/**
 * @brief 按原始轮廓编号查找轮廓信息。
 * @param result ActiveFanDetector 的检测结果。
 * @param id 需要查找的轮廓编号。
 * @return 找到时返回轮廓地址，否则返回 nullptr。
 */
inline const ActiveFanDetector::IndexedContour * find_contour(
  const ActiveFanDetector::DetectionResult & result, int id)
{
  for (const auto & contour : result.detected_contours) {
    if (contour.id == id) return &contour;
  }
  return nullptr;
}

/**
 * @brief 绘制单个轮廓，并在其中心标注原始轮廓编号。
 * @param image 输出可视化图像。
 * @param contour 待绘制的已编号轮廓。
 * @param color 轮廓及编号使用的 BGR 颜色。
 * @param thickness 轮廓线宽。
 */
inline void draw_contour(
  cv::Mat & image, const ActiveFanDetector::IndexedContour & contour,
  const cv::Scalar & color, int thickness)
{
  cv::polylines(image, contour.points, true, color, thickness, cv::LINE_AA);
  cv::putText(
    image, std::to_string(contour.id), contour.center, cv::FONT_HERSHEY_SIMPLEX, 0.5,
    color, 1, cv::LINE_AA);
}

/**
 * @brief 按状态绘制 big ROI、已剔除轮廓和最终保留轮廓。
 * @param image 输出可视化图像。
 * @param result ActiveFanDetector 的完整检测结果。
 */
inline void draw_detection_result(
  cv::Mat & image, const ActiveFanDetector::DetectionResult & result)
{
  for (const auto & contour : result.detected_contours) {
    draw_contour(image, contour, {128, 128, 128}, 1);
  }

  const cv::Rect big_roi = integer_rect(result.big_roi, image.size());
  if (!big_roi.empty()) cv::rectangle(image, big_roi, {255, 255, 0}, 2, cv::LINE_AA);

  for (int id : result.outside_big_roi_ids) {
    const auto * contour = find_contour(result, id);
    if (contour != nullptr) draw_contour(image, *contour, {0, 0, 255}, 2);
  }
  for (int id : result.smaller_than_rune_center_ids) {
    const auto * contour = find_contour(result, id);
    if (contour != nullptr) draw_contour(image, *contour, {0, 165, 255}, 2);
  }
  for (int id : result.roi_excluded_ids) {
    const auto * contour = find_contour(result, id);
    if (contour != nullptr) draw_contour(image, *contour, {255, 0, 255}, 2);
  }
  for (const auto & contour : result.remaining_contours) {
    draw_contour(image, contour, {0, 255, 0}, 2);
  }
}

/**
 * @brief 以方括号列表形式向标准输出打印轮廓编号。
 * @param ids 待打印的轮廓编号集合。
 */
inline void print_ids(const std::vector<int> & ids)
{
  std::cout << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << ']';
}

/**
 * @brief 输出当前帧 big ROI、各深度 ROI 和最终保留轮廓的编号名单。
 * @param frame_count 当前视频帧编号。
 * @param objects YOLO 输出的全部识别对象。
 * @param roi_object_indexes ROI 下标到 YOLO 对象下标的映射。
 * @param result ActiveFanDetector 的完整检测结果。
 */
inline void print_exclusion_list(
  int frame_count, const std::vector<auto_buff::YOLO11_BUFF::Object> & objects,
  const std::vector<int> & roi_object_indexes,
  const ActiveFanDetector::DetectionResult & result)
{
  std::cout << "[frame " << frame_count << "] big_roi outside contours = ";
  print_ids(result.outside_big_roi_ids);
  std::cout << '\n';
  std::cout << "  rune_center contour[" << result.rune_center_contour_id
            << "] area=" << result.rune_center_contour_area
            << ", smaller contours = ";
  print_ids(result.smaller_than_rune_center_ids);
  std::cout << '\n';

  for (std::size_t i = 0; i < roi_object_indexes.size(); ++i) {
    const int object_index = roi_object_indexes[i];
    const auto & object = objects[static_cast<std::size_t>(object_index)];
    std::cout << "  roi[" << i << "] object[" << object_index << "] "
              << label_name(object.label) << " conf=" << cv::format("%.2f", object.prob)
              << " -> contour[" << result.roi_excluded_ids[i] << "]\n";
  }
  std::cout << "  remaining contours = ";
  std::vector<int> remaining_ids;
  remaining_ids.reserve(result.remaining_contours.size());
  for (const auto & contour : result.remaining_contours) remaining_ids.push_back(contour.id);
  print_ids(remaining_ids);
  std::cout << '\n';
}

}  // namespace auto_buff::big
