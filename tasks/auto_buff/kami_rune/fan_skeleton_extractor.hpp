#pragma once

/**
 * @file fan_skeleton_extractor.hpp
 * @brief 已激活扇叶灯条正反边界配对与骨架恢复的公共接口。
 *
 * 本文件定义算法参数、各阶段中间结果以及骨架提取器。调用方既可以输入完整
 * 二值图，也可以绕过候选筛选，直接分析已经选中的闭合轮廓。
 */

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace auto_buff::kami_rune
{

/** @brief 骨架提取全流程参数。所有长度参数单位均为像素，角度参数单位均为度。 */
struct FanSkeletonParams
{
  // 第一阶段：轮廓候选筛选参数。
  int min_contour_points = 30;  //!< 轮廓最少像素点数。
  double min_contour_area = 30.0;  //!< 轮廓面积下限。
  double max_contour_area = std::numeric_limits<double>::infinity();  //!< 轮廓面积上限。
  double max_contour_aspect_ratio = 3.0;  //!< 最小外接矩形最大长短边比。
  double min_contour_fill_ratio = 0.03;  //!< 轮廓占最小外接矩形的面积比例下限。
  double max_contour_fill_ratio = 0.85;  //!< 轮廓占最小外接矩形的面积比例上限。
  double min_contour_compactness = 0.0001;  //!< 面积除以周长平方的下限。
  double max_contour_compactness = 0.08;  //!< 面积除以周长平方的上限。
  bool external_contours_only = true;  //!< 是否只提取最外层轮廓。

  // 第二阶段：闭合方向信号平滑参数。
  double smoothing_window_ratio = 0.04;  //!< 高斯窗口长度占轮廓点数的比例。
  int min_smoothing_window = 5;  //!< 高斯窗口最小长度，最终会调整为奇数。
  int max_smoothing_window = 101;  //!< 高斯窗口最大长度，最终会调整为奇数。
  double smoothing_sigma = 3.0;  //!< 高斯核标准差。

  // 第三阶段：方向稳定区间和直线边界拟合参数。
  double max_tangent_gradient_deg = 5.0;  //!< 判定方向稳定的最大局部角度梯度。
  int max_straight_gap_samples = 12;  //!< 可填补的不稳定短缺口最大点数。
  double max_straight_gap_ratio = 0.015;  //!< 短缺口最大长度占轮廓点数的比例。
  double max_straight_merge_angle_deg = 8.0;  //!< 缺口两端方向允许的最大夹角。
  int min_segment_samples = 6;  //!< 单条候选边界最少采样点数。
  double min_segment_samples_ratio = 0.008;  //!< 边界最少点数占轮廓点数的比例。
  double max_segment_rms = 2.5;  //!< 直线拟合均方根误差的绝对上限。
  double max_segment_rms_ratio = 0.04;  //!< 拟合误差相对边界长度的上限。

  // 第四阶段：正反边界配对参数。
  double opposite_angle_tolerance_deg = 24.0;  //!< 边界方向偏离 180 度的最大误差。
  double max_pair_arc_gap_ratio = 0.38;  //!< 两边界沿闭合轮廓最短间隔比例上限。
  double min_strip_width = 0.5;  //!< 灯条最小宽度。
  double max_strip_width_bbox_ratio = 0.55;  //!< 最大宽度占轮廓包围盒对角线的比例。
  double min_pair_overlap_ratio = 0.20;  //!< 两边界沿骨架方向的最小投影重叠率。
  double min_pair_aspect_ratio = 1.5;  //!< 边界长度与灯条宽度的最小比例。
  double max_pair_longitudinal_offset_ratio = 0.65;  //!< 两边界中心最大轴向错位比例。
};

/** @brief 一个闭合轮廓转换得到的一维方向信号。 */
struct DirectionSignal
{
  std::vector<cv::Point2f> raw_directions;  //!< 相邻轮廓点差分得到的单位方向。
  std::vector<cv::Point2f> smoothed_directions;  //!< 使用圆周高斯滤波后的单位方向。
  std::vector<float> angles_deg;  //!< 平滑方向映射到的角度。
  std::vector<float> gradient_deg;  //!< 角度沿轮廓的局部变化率。
};

/** @brief 从一个方向稳定区间拟合出的有向直线边界。 */
struct BoundarySegment
{
  int start_index = -1;  //!< 稳定区间在原轮廓中的起始下标。
  int end_index = -1;  //!< 稳定区间在原轮廓中的结束下标。
  int sample_count = 0;  //!< 参与稳定区间判定的采样点数。
  bool wraps_contour_end = false;  //!< 区间是否跨过轮廓数组首尾。
  cv::Point2f center{};  //!< 拟合点集的平均中心。
  cv::Point2f direction{};  //!< 与轮廓遍历方向一致的单位方向。
  cv::Point2f start_point{};  //!< 所有拟合点投影后的边界起点。
  cv::Point2f end_point{};  //!< 所有拟合点投影后的边界终点。
  float angle_deg = 0.0F;  //!< 有向边界角度。
  float length = 0.0F;  //!< 投影起止点之间的长度。
  float rms_error = 0.0F;  //!< 点到拟合直线的均方根距离。
};

/** @brief 一对正反边界中间恢复出的灯条骨架线段。 */
struct SkeletonLine
{
  cv::Point2f start{};  //!< 两边界公共投影区间的起点。
  cv::Point2f end{};  //!< 两边界公共投影区间的终点。
  cv::Point2f center{};  //!< 骨架中点。
  cv::Point2f direction{};  //!< 骨架单位方向；正负方向只由轮廓遍历顺序决定。
  float length = 0.0F;  //!< 骨架有效支撑长度。
};

/** @brief 两条反向边界的匹配结果及其质量信息。 */
struct BoundaryPair
{
  int first_boundary = -1;  //!< 第一条边界在结果数组中的下标。
  int second_boundary = -1;  //!< 第二条边界在结果数组中的下标。
  SkeletonLine skeleton;  //!< 从边界公共投影区间恢复的骨架。
  float strip_width = 0.0F;  //!< 两边界的法向距离。
  float opposite_angle_error_deg = 0.0F;  //!< 方向夹角相对 180 度的误差。
  float overlap_ratio = 0.0F;  //!< 公共投影长度除以较短边界长度。
  float score = std::numeric_limits<float>::infinity();  //!< 综合代价，越小越可信。
};

/** @brief 单个闭合轮廓经过全流程后的结果。 */
struct ContourSkeletonResult
{
  int contour_index = -1;  //!< 轮廓在输入集合中的编号，未知时为 -1。
  std::vector<cv::Point> contour;  //!< 原始完整闭合轮廓。
  DirectionSignal signal;  //!< 轮廓方向信号，保留用于调试和可视化。
  std::vector<BoundarySegment> boundaries;  //!< 方向稳定区间拟合出的全部边界。
  std::vector<BoundaryPair> pairs;  //!< 一对一筛选后的正反边界及骨架。
};

/** @brief 已激活扇叶灯条骨架提取器。 */
class FanSkeletonExtractor
{
public:
  /** @brief 使用给定参数创建提取器，并检查关键参数是否合法。 */
  explicit FanSkeletonExtractor(FanSkeletonParams params = {});

  /** @brief 返回当前只读参数。 */
  const FanSkeletonParams & params() const noexcept;

  /**
   * @brief 从二值图执行完整流程。
   * @param binary 非空的 CV_8UC1 二值图。
   * @return 通过候选筛选的所有轮廓及其骨架结果。
   * @throws std::invalid_argument 输入图像为空或类型不正确。
   */
  std::vector<ContourSkeletonResult> extract(const cv::Mat & binary) const;

  /**
   * @brief 跳过候选筛选，直接分析一个已选中的闭合轮廓。
   * @param contour 按边界遍历顺序排列的完整轮廓点。
   * @param contour_index 可选的外部轮廓编号。
   */
  ContourSkeletonResult analyze_contour(
    const std::vector<cv::Point> & contour, int contour_index = -1) const;

private:
  /** @brief 使用面积、外接矩形和紧致度判断轮廓是否值得继续处理。 */
  bool is_candidate(const std::vector<cv::Point> & contour) const;
  /** @brief 将二维闭合轮廓转换为平滑方向及局部角度梯度信号。 */
  DirectionSignal build_direction_signal(const std::vector<cv::Point> & contour) const;
  /** @brief 将低梯度连续区间压缩为经过拟合的有向直线边界。 */
  std::vector<BoundarySegment> extract_boundaries(
    const std::vector<cv::Point> & contour, const DirectionSignal & signal) const;
  /** @brief 筛选方向相反的边界，并恢复一对一灯条骨架。 */
  std::vector<BoundaryPair> match_boundaries(
    const std::vector<cv::Point> & contour,
    const std::vector<BoundarySegment> & boundaries) const;

  FanSkeletonParams params_;  //!< 当前提取器使用的参数副本。
};


/** @brief 可视化测试接口返回的骨架特征点。 */
struct SkeletonFeaturePoints
{
  std::vector<SkeletonLine> skeletons;  //!< 当前帧检测并绘制的骨架。
  std::vector<cv::Point2f> endpoints;  //!< 所有骨架线段端点，按像素距离去重。
  std::vector<cv::Point2f> corners;  //!< 不同骨架线段相交形成的拐角，按像素距离去重。
  cv::Mat visualization;  //!< 已绘制轮廓、骨架、端点和拐角的 CV_8UC3 图像。
};

/** @brief 基于闭合轮廓方向梯度的灯条端点提取参数。 */
struct GradientEndpointParams
{
  float min_gradient_deg = 4.0F;  //!< 低通后凸响应的绝对下限。
  float relative_gradient_ratio = 0.35F;  //!< 区域阈值占本轮廓最大凸响应的比例。
  int convex_smoothing_radius = 12;  //!< 凸响应低通滤波的轮廓采样半径。
  float convex_smoothing_sigma = 6.0F;  //!< 凸响应高斯低通标准差。
  int local_max_radius = 3;  //!< 区域内寻找主峰时左右检查的轮廓采样半径。
  int min_tip_arc_samples = 2;  //!< 粗灯条两个端帽直角峰之间的最少采样点数。
  int min_tip_arc_limit_samples = 12;  //!< 合并同一凸起内短缺口的绝对长度上限。
  float max_tip_arc_ratio = 0.10F;  //!< 合并缺口长度占轮廓点数的比例上限。
  float min_peak_merge_distance = 8.0F;  //!< 同一凸起两侧允许的最小空间距离上限。
  float peak_merge_distance_bbox_ratio = 0.20F;  //!< 空间距离上限占包围盒对角线的比例。
  float max_double_corner_valley_ratio = 0.35F;  //!< 双直角峰之间低谷相对较弱峰的上限。
  float endpoint_merge_distance = 4.0F;  //!< 不同凸起生成端点后的去重距离。
  float min_unpaired_peak_ratio = 0.50F;  //!< 第二直角峰相对区域主峰的强度下限。
  int image_border_margin = 2;  //!< 距图像边界小于该值的峰值不视为真实端点。
  int max_endpoints_per_contour = 8;  //!< 单个轮廓最多返回的端点数。
};

/** @brief 轮廓梯度端点检测与绘制结果。 */
struct GradientEndpointResult
{
  std::vector<cv::Point2f> endpoints;  //!< 当前帧所有轮廓提取到的端帽中心。
  cv::Mat visualization;  //!< 已绘制绿色轮廓和红色端点的 CV_8UC3 图像。
};

/**
 * @brief 从剔除后的闭合轮廓中为每个凸起区域提取唯一端点。
 * @param binary 非空的 CV_8UC1 二值图。
 * @param contours 已完成 ROI 剔除的闭合轮廓集合。
 * @param skeleton_params 方向信号使用的平滑参数。
 * @param endpoint_params 凸响应平滑、区域合并和端点过滤参数。
 * @param window_name 显示窗口名称；为空时不调用 imshow。
 */
GradientEndpointResult detect_and_show_gradient_endpoints(
  const cv::Mat & binary, const std::vector<std::vector<cv::Point>> & contours,
  const FanSkeletonParams & skeleton_params = {},
  const GradientEndpointParams & endpoint_params = {},
  const std::string & window_name = "kami_rune gradient endpoints");

/** @brief 支持直接传入 detection.remaining_contours 等带 points 字段的轮廓集合。 */
template<typename IndexedContour>
GradientEndpointResult detect_and_show_gradient_endpoints(
  const cv::Mat & binary, const std::vector<IndexedContour> & indexed_contours,
  const FanSkeletonParams & skeleton_params = {},
  const GradientEndpointParams & endpoint_params = {},
  const std::string & window_name = "kami_rune gradient endpoints")
{
  std::vector<std::vector<cv::Point>> contours;
  contours.reserve(indexed_contours.size());
  for (const IndexedContour & contour : indexed_contours) contours.push_back(contour.points);
  return detect_and_show_gradient_endpoints(
    binary, contours, skeleton_params, endpoint_params, window_name);
}

/**
 * @brief 分析给定的剔除后轮廓集，显示轮廓、骨架、端点和骨架交点。
 * @param binary 非空的 CV_8UC1 二值图。
 * @param contours 已完成 ROI 剔除的闭合轮廓点集合。
 * @param params 骨架提取参数，可按真实图像尺度调整。
 * @param window_name OpenCV 显示窗口名称；为空时只返回可视化图，不调用 imshow。
 * @return 去重后的骨架端点、交点和已绘制的 BGR 可视化图。
 *
 * 可视化颜色：绿色为原轮廓，黄色为骨架，深蓝色圆点为端点和拐角；
 * 端点使用 E 编号，拐角使用 C 编号。
 * 拐角定义为两条非平行骨架在线段范围附近的交点，而不是原始像素轮廓拐点。
 * 本函数只负责 imshow，按键和刷新应由调用方所在的事件循环执行。
 */
SkeletonFeaturePoints detect_and_show_skeleton(
  const cv::Mat & binary, const std::vector<std::vector<cv::Point>> & contours,
  const FanSkeletonParams & params = {},
  const std::string & window_name = "kami_rune skeleton");

/**
 * @brief 接收 ActiveFanDetector::IndexedContour 一类带 points 字段的轮廓集合。
 *
 * 该重载使调用方可以直接传入 detection.remaining_contours，同时保持骨架模块
 * 不依赖 ActiveFanDetector 的具体类型。
 */
template<typename IndexedContour>
SkeletonFeaturePoints detect_and_show_skeleton(
  const cv::Mat & binary, const std::vector<IndexedContour> & indexed_contours,
  const FanSkeletonParams & params = {},
  const std::string & window_name = "kami_rune skeleton")
{
  std::vector<std::vector<cv::Point>> contours;
  contours.reserve(indexed_contours.size());
  for (const IndexedContour & contour : indexed_contours) contours.push_back(contour.points);
  return detect_and_show_skeleton(binary, contours, params, window_name);
}

}  // namespace auto_buff::kami_rune
