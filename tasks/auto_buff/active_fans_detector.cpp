#include "active_fans_detector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace auto_buff::big
{
namespace
{
/**
 * @brief 将字符串转换为小写，用于兼容配置项的大小写写法。
 */
std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

/**
 * @brief 计算轮廓质心；退化轮廓无法通过矩计算时使用外接矩形中心。
 */
cv::Point2f contour_center(const ActiveFanDetector::Contour & contour)
{
  const cv::Moments moments = cv::moments(contour);
  if (std::abs(moments.m00) > std::numeric_limits<double>::epsilon()) {
    return {
      static_cast<float>(moments.m10 / moments.m00),
      static_cast<float>(moments.m01 / moments.m00)};
  }

  const cv::Rect bounds = cv::boundingRect(contour);
  return {
    bounds.x + bounds.width * 0.5f,
    bounds.y + bounds.height * 0.5f};
}

/**
 * @brief 将 ROI 裁剪到图像边界内，避免后续出现越界区域。
 */
ActiveFanDetector::Roi clip_roi(
  const ActiveFanDetector::Roi & roi, const cv::Size & image_size)
{
  return roi & ActiveFanDetector::Roi(
                 0.0f, 0.0f, static_cast<float>(image_size.width),
                 static_cast<float>(image_size.height));
}

/**
 * @brief 以 ROI 中心为基准按指定倍数放大，并将结果限制在图像范围内。
 */
ActiveFanDetector::Roi scale_roi(
  const ActiveFanDetector::Roi & roi, float scale, const cv::Size & image_size)
{
  const cv::Point2f center(
    roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
  const float width = roi.width * scale;
  const float height = roi.height * scale;
  return clip_roi(
    {center.x - width * 0.5f, center.y - height * 0.5f, width, height}, image_size);
}

/**
 * @brief 优先从 active_fan_detector 子节点读取配置，不存在时回退到根节点。
 */
const YAML::Node find_node(
  const YAML::Node & primary, const YAML::Node & root, const std::string & key)
{
  if (primary[key]) return primary[key];
  return root[key];
}
}  // namespace

/**
 * @brief 从 YAML 文件读取 ROI 放大倍数、目标颜色、颜色阈值和轮廓选择策略。
 */
ActiveFanDetector::ActiveFanDetector(const std::string & config)
{
  const YAML::Node root = YAML::LoadFile(config);
  const YAML::Node node = root["active_fan_detector"] ? root["active_fan_detector"] : root;

  if (const auto value = find_node(node, root, "roi_scale")) roi_scale_ = value.as<float>();
  if (!std::isfinite(roi_scale_) || roi_scale_ <= 0.0f) {
    throw std::invalid_argument("active fan roi_scale must be greater than zero");
  }

  YAML::Node color_node = find_node(node, root, "buff_color");
  if (!color_node) color_node = find_node(node, root, "enemy_color");
  if (color_node) {
    const std::string value = lower(color_node.Scalar());
    if (value == "red" || value == "false" || value == "0") {
      color_ = PixChannel::RED;
    } else if (value == "blue" || value == "true" || value == "1" || value == "auto") {
      color_ = PixChannel::BLUE;
    } else {
      throw std::invalid_argument("active fan buff_color must be red or blue");
    }
  }

  YAML::Node threshold_node = find_node(node, root, "color_thresh");
  if (!threshold_node) threshold_node = find_node(node, root, "color_threshold");
  if (threshold_node) {
    const int threshold = threshold_node.as<int>();
    if (threshold < 0 || threshold > 255) {
      throw std::invalid_argument("active fan color threshold must be in [0, 255]");
    }
    color_thresh_ = static_cast<std::uint8_t>(threshold);
  }

  if (const auto selection_node = find_node(node, root, "roi_contour_selection")) {
    const std::string selection = lower(selection_node.as<std::string>());
    if (selection == "largest_area") {
      roi_contour_selection_ = RoiContourSelection::LARGEST_AREA;
    } else if (selection == "nearest_center") {
      roi_contour_selection_ = RoiContourSelection::NEAREST_CENTER;
    } else {
      throw std::invalid_argument(
        "active fan roi_contour_selection must be largest_area or nearest_center");
    }
  }
}

/**
 * @brief 计算目标颜色通道与相反颜色通道的差值，并按颜色阈值生成二值图。
 */
cv::Mat ActiveFanDetector::binarize(const cv::Mat & bgr_img) const
{
  if (bgr_img.empty()) return {};
  if (bgr_img.type() != CV_8UC3) {
    throw std::invalid_argument("ActiveFanDetector expects a CV_8UC3 BGR image");
  }

  cv::Mat target_channel;
  cv::Mat opposite_channel;
  cv::extractChannel(bgr_img, target_channel, static_cast<int>(color_));
  cv::extractChannel(
    bgr_img, opposite_channel,
    static_cast<int>(color_ == PixChannel::RED ? PixChannel::BLUE : PixChannel::RED));

  cv::Mat color_difference;
  cv::subtract(target_channel, opposite_channel, color_difference);

  cv::Mat binary;
  cv::threshold(color_difference, binary, color_thresh_, 255, cv::THRESH_BINARY);
  return binary;
}

/**
 * @brief 提取并编号轮廓，依次剔除 big ROI 外轮廓和各外部识别框对应的轮廓。
 *
 * 每个轮廓以质心判断所属区域。外部识别框按输入顺序处理，每个框根据配置选择
 * 面积最大或最靠近框中心的一个轮廓；已经剔除的轮廓不会被后续框重复选择。
 */
ActiveFanDetector::DetectionResult ActiveFanDetector::detect(
  const cv::Mat & bgr_img, const Roi & r_roi, const std::vector<Roi> & rois) const
{
  DetectionResult result;
  result.roi_excluded_ids.assign(rois.size(), -1);
  if (bgr_img.empty()) return result;
  if (r_roi.width <= 0.0f || r_roi.height <= 0.0f) {
    throw std::invalid_argument("ActiveFanDetector r_roi must not be empty");
  }

  result.binary = binarize(bgr_img);
  result.big_roi = scale_roi(r_roi, roi_scale_, bgr_img.size());

  // 从整张二值图提取最外层轮廓，避免孔洞轮廓被重复编号。
  std::vector<Contour> contours;
  cv::findContours(result.binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::imshow("ActiveFanDetector binary", result.binary);

  // 按 findContours 的输出顺序编号，并剔除质心位于 big ROI 外的轮廓。
  result.detected_contours.reserve(contours.size());
  std::vector<bool> keep(contours.size(), true);
  for (std::size_t i = 0; i < contours.size(); ++i) {
    IndexedContour indexed;
    indexed.id = static_cast<int>(i);
    indexed.points = std::move(contours[i]);
    indexed.area = cv::contourArea(indexed.points);
    indexed.center = contour_center(indexed.points);
    result.detected_contours.push_back(std::move(indexed));

    if (!result.big_roi.contains(result.detected_contours.back().center)) {
      keep[i] = false;
      result.outside_big_roi_ids.push_back(static_cast<int>(i));
    }
  }

  // 每个外部识别框只选择并剔除一个当前仍保留的轮廓。
  for (std::size_t roi_index = 0; roi_index < rois.size(); ++roi_index) {
    const Roi roi = clip_roi(rois[roi_index], bgr_img.size());
    if (roi.width <= 0.0f || roi.height <= 0.0f) continue;

    const cv::Point2f roi_center(
      roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
    int selected_id = -1;
    double best_area = -1.0;
    double best_distance = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < result.detected_contours.size(); ++i) {
      if (!keep[i]) continue;
      const IndexedContour & contour = result.detected_contours[i];
      if (!roi.contains(contour.center)) continue;

      const double distance = cv::norm(contour.center - roi_center);
      const bool prefer_larger =
        contour.area > best_area || (contour.area == best_area && distance < best_distance);
      const bool prefer_nearer =
        distance < best_distance || (distance == best_distance && contour.area > best_area);
      const bool better = roi_contour_selection_ == RoiContourSelection::LARGEST_AREA
                            ? prefer_larger
                            : prefer_nearer;
      if (!better) continue;

      selected_id = contour.id;
      best_area = contour.area;
      best_distance = distance;
    }

    result.roi_excluded_ids[roi_index] = selected_id;
    if (selected_id >= 0) keep[static_cast<std::size_t>(selected_id)] = false;
  }

  // 根据保留标记汇总最终可用于后续检测的轮廓，同时保持原始编号不变。
  for (std::size_t i = 0; i < result.detected_contours.size(); ++i) {
    if (keep[i]) result.remaining_contours.push_back(result.detected_contours[i]);
  }
  return result;
}
}  // namespace auto_buff::big
