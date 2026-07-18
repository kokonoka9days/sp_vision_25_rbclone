#include "fan_skeleton_extractor.hpp"

/**
 * @file fan_skeleton_extractor.cpp
 * @brief 已激活扇叶轮廓方向信号、正反边界配对和灯条骨架恢复实现。
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace auto_buff::kami_rune
{
namespace
{

constexpr float kEpsilon = 1e-6F;
constexpr float kRadToDeg = 180.0F / static_cast<float>(CV_PI);

// 将任意整数下标映射到 [0, size)，用于把轮廓数组当作周期数据访问。
int circular_index(int index, int size)
{
  index %= size;
  return index < 0 ? index + size : index;
}

// 安全归一化二维向量，零向量保持为零，避免后续产生 NaN。
cv::Point2f normalized(const cv::Point2f & value)
{
  const float magnitude = std::sqrt(value.dot(value));
  return magnitude > kEpsilon ? value / magnitude : cv::Point2f{};
}

// 使用叉积和点积直接计算有符号圆周角，结果位于 [-180, 180]。
float signed_angle_deg(const cv::Point2f & from, const cv::Point2f & to)
{
  return std::atan2(from.cross(to), from.dot(to)) * kRadToDeg;
}

// 返回两个单位方向之间 [0, 180] 范围内的夹角。
float undirected_angle_deg(const cv::Point2f & first, const cv::Point2f & second)
{
  const float cosine = std::clamp(first.dot(second), -1.0F, 1.0F);
  return std::acos(cosine) * kRadToDeg;
}

// 约束滤波窗口范围，并保证窗口为不超过轮廓长度的正奇数。
int odd_clamped_window(int requested, int minimum, int maximum, int point_count)
{
  int result = std::clamp(requested, minimum, maximum);
  result = std::min(result, point_count % 2 == 0 ? point_count - 1 : point_count);
  if (result < 1) return 1;
  return result % 2 == 0 ? result - 1 : result;
}

// 构造归一化的一维高斯卷积核；sigma 为零时根据窗口长度自动估计。
std::vector<float> gaussian_kernel(int window, double sigma)
{
  std::vector<float> kernel(static_cast<std::size_t>(window));
  const int radius = window / 2;
  const double effective_sigma = sigma > 0.0 ? sigma : std::max(1.0, window / 6.0);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const double weight = std::exp(
      -static_cast<double>(i * i) / (2.0 * effective_sigma * effective_sigma));
    kernel[static_cast<std::size_t>(i + radius)] = static_cast<float>(weight);
    sum += weight;
  }
  for (float & weight : kernel) weight = static_cast<float>(weight / sum);
  return kernel;
}

/**
 * @brief 填补方向稳定区间中的短暂断点。
 *
 * 只有缺口长度和两端方向差同时满足阈值时才填补，避免把拐角两侧的不同
 * 直线误合并。遍历和填补均按闭合轮廓处理，因此首尾处没有特殊断点。
 */
void close_short_unstable_gaps(
  std::vector<unsigned char> & stable, const std::vector<cv::Point2f> & directions,
  int max_gap, double max_angle_deg)
{
  const int size = static_cast<int>(stable.size());
  const auto stable_it = std::find(stable.begin(), stable.end(), 1U);
  if (stable_it == stable.end()) return;

  const int seed = static_cast<int>(std::distance(stable.begin(), stable_it));
  int visited = 0;
  int index = (seed + 1) % size;
  while (visited < size) {
    if (stable[static_cast<std::size_t>(index)] != 0U) {
      index = (index + 1) % size;
      ++visited;
      continue;
    }

    const int gap_start = index;
    int gap_length = 0;
    while (visited + gap_length < size && stable[static_cast<std::size_t>(index)] == 0U) {
      index = (index + 1) % size;
      ++gap_length;
    }
    visited += gap_length;

    const int left = circular_index(gap_start - 1, size);
    const int right = index;
    if (gap_length <= max_gap && stable[static_cast<std::size_t>(right)] != 0U &&
        std::abs(signed_angle_deg(directions[static_cast<std::size_t>(left)],
                                  directions[static_cast<std::size_t>(right)])) <=
          max_angle_deg) {
      for (int i = 0; i < gap_length; ++i) {
        stable[static_cast<std::size_t>(circular_index(gap_start + i, size))] = 1U;
      }
    }
  }
}

// 从闭合布尔掩码中提取连续真值区间，每个区间保存对应的原轮廓下标。
std::vector<std::vector<int>> circular_true_runs(const std::vector<unsigned char> & mask)
{
  std::vector<std::vector<int>> runs;
  const int size = static_cast<int>(mask.size());
  const auto false_it = std::find(mask.begin(), mask.end(), 0U);
  if (false_it == mask.end()) {
    std::vector<int> all(static_cast<std::size_t>(size));
    std::iota(all.begin(), all.end(), 0);
    runs.push_back(std::move(all));
    return runs;
  }

  const int first_false = static_cast<int>(std::distance(mask.begin(), false_it));
  int visited = 0;
  int index = (first_false + 1) % size;
  while (visited < size) {
    if (mask[static_cast<std::size_t>(index)] == 0U) {
      index = (index + 1) % size;
      ++visited;
      continue;
    }
    std::vector<int> run;
    while (visited < size && mask[static_cast<std::size_t>(index)] != 0U) {
      run.push_back(index);
      index = (index + 1) % size;
      ++visited;
    }
    runs.push_back(std::move(run));
  }
  return runs;
}

/**
 * @brief 对一个稳定方向区间进行直线拟合并生成边界描述。
 *
 * cv::fitLine 给出的方向本身没有正负含义，因此这里使用平滑轮廓方向校正
 * 正负，使一对灯条边界仍保留“遍历方向相反”这一关键性质。
 */
BoundarySegment fit_boundary(
  const std::vector<cv::Point> & contour, const std::vector<int> & indices,
  const DirectionSignal & signal)
{
  std::vector<cv::Point2f> points;
  points.reserve(indices.size() + 1U);
  cv::Point2f average_direction{};
  for (int index : indices) {
    points.emplace_back(contour[static_cast<std::size_t>(index)]);
    average_direction += signal.smoothed_directions[static_cast<std::size_t>(index)];
  }
  points.emplace_back(contour[static_cast<std::size_t>((indices.back() + 1) % contour.size())]);

  cv::Vec4f line;
  cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
  cv::Point2f direction = normalized({line[0], line[1]});
  if (direction.dot(average_direction) < 0.0F) direction *= -1.0F;

  // 使用点集平均位置作为投影原点，降低全局坐标数值对计算的影响。
  cv::Point2f center{};
  for (const cv::Point2f & point : points) center += point;
  center /= static_cast<float>(points.size());

  // 沿拟合方向的投影极值给出有效端点，法向残差用于衡量直线质量。
  float min_projection = std::numeric_limits<float>::infinity();
  float max_projection = -std::numeric_limits<float>::infinity();
  double squared_error = 0.0;
  const cv::Point2f normal{-direction.y, direction.x};
  for (const cv::Point2f & point : points) {
    const cv::Point2f offset = point - center;
    const float projection = offset.dot(direction);
    min_projection = std::min(min_projection, projection);
    max_projection = std::max(max_projection, projection);
    const float residual = offset.dot(normal);
    squared_error += static_cast<double>(residual * residual);
  }

  BoundarySegment result;
  result.start_index = indices.front();
  result.end_index = indices.back();
  result.sample_count = static_cast<int>(indices.size());
  result.wraps_contour_end = result.end_index < result.start_index;
  result.center = center;
  result.direction = direction;
  result.start_point = center + min_projection * direction;
  result.end_point = center + max_projection * direction;
  result.angle_deg = std::atan2(direction.y, direction.x) * kRadToDeg;
  result.length = max_projection - min_projection;
  result.rms_error = static_cast<float>(std::sqrt(squared_error / points.size()));
  return result;
}

// 计算从一条边界结束位置沿轮廓正向走到另一条边界起点的采样间隔。
int forward_gap(const BoundarySegment & from, const BoundarySegment & to, int contour_size)
{
  return circular_index(to.start_index - from.end_index - 1, contour_size);
}

// 内部候选结构额外保存边界下标，便于排序后执行一对一选择。
struct PairCandidate
{
  int first = -1;
  int second = -1;
  BoundaryPair pair;
};

}  // namespace

FanSkeletonExtractor::FanSkeletonExtractor(FanSkeletonParams params) : params_(std::move(params))
{
  // 提前拒绝会破坏数组访问或几何含义的参数，其余阈值允许调用方按数据调整。
  if (params_.min_contour_points < 3) {
    throw std::invalid_argument("min_contour_points must be at least 3");
  }
  if (params_.smoothing_window_ratio < 0.0 || params_.smoothing_sigma < 0.0) {
    throw std::invalid_argument("smoothing parameters must be non-negative");
  }
  if (params_.opposite_angle_tolerance_deg <= 0.0 ||
      params_.opposite_angle_tolerance_deg >= 90.0) {
    throw std::invalid_argument("opposite_angle_tolerance_deg must be in (0, 90)");
  }
}

const FanSkeletonParams & FanSkeletonExtractor::params() const noexcept { return params_; }

std::vector<ContourSkeletonResult> FanSkeletonExtractor::extract(const cv::Mat & binary) const
{
  if (binary.empty() || binary.type() != CV_8UC1) {
    throw std::invalid_argument("binary image must be a non-empty CV_8UC1 matrix");
  }

  // findContours 可能修改输入图，因此使用副本；CHAIN_APPROX_NONE 保留全部边界像素。
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat work = binary.clone();
  cv::findContours(
    work, contours, hierarchy,
    params_.external_contours_only ? cv::RETR_EXTERNAL : cv::RETR_TREE,
    cv::CHAIN_APPROX_NONE);

  std::vector<ContourSkeletonResult> results;
  for (std::size_t index = 0; index < contours.size(); ++index) {
    if (!is_candidate(contours[index])) continue;
    results.push_back(analyze_contour(contours[index], static_cast<int>(index)));
  }
  return results;
}

ContourSkeletonResult FanSkeletonExtractor::analyze_contour(
  const std::vector<cv::Point> & contour, int contour_index) const
{
  if (contour.size() < 3U) throw std::invalid_argument("contour must contain at least 3 points");

  // 明确保存各阶段结果，既方便上层继续求交，也方便逐阶段可视化排查。
  ContourSkeletonResult result;
  result.contour_index = contour_index;
  result.contour = contour;
  result.signal = build_direction_signal(contour);
  result.boundaries = extract_boundaries(contour, result.signal);
  result.pairs = match_boundaries(contour, result.boundaries);
  return result;
}

bool FanSkeletonExtractor::is_candidate(const std::vector<cv::Point> & contour) const
{
  // 点数和面积是成本最低的检查，优先执行以减少后续几何计算。
  if (static_cast<int>(contour.size()) < params_.min_contour_points) return false;
  const double area = std::abs(cv::contourArea(contour));
  if (area < params_.min_contour_area || area > params_.max_contour_area) return false;

  // 外接矩形同时提供整体长宽比和填充率，排除极端细长或近似实心矩形目标。
  const cv::RotatedRect rectangle = cv::minAreaRect(contour);
  const double long_side = std::max(rectangle.size.width, rectangle.size.height);
  const double short_side = std::min(rectangle.size.width, rectangle.size.height);
  if (short_side <= kEpsilon || long_side / short_side > params_.max_contour_aspect_ratio) {
    return false;
  }

  const double rectangle_area = rectangle.size.area();
  const double fill_ratio = rectangle_area > 0.0 ? area / rectangle_area : 0.0;
  if (fill_ratio < params_.min_contour_fill_ratio ||
      fill_ratio > params_.max_contour_fill_ratio) {
    return false;
  }

  // 面积/周长平方具有尺度不变性，可进一步抑制毛刺严重或过于紧凑的轮廓。
  const double perimeter = cv::arcLength(contour, true);
  const double compactness = perimeter > 0.0 ? area / (perimeter * perimeter) : 0.0;
  return compactness >= params_.min_contour_compactness &&
         compactness <= params_.max_contour_compactness;
}

DirectionSignal FanSkeletonExtractor::build_direction_signal(
  const std::vector<cv::Point> & contour) const
{
  const int size = static_cast<int>(contour.size());
  DirectionSignal signal;
  signal.raw_directions.resize(contour.size());
  signal.smoothed_directions.resize(contour.size());
  signal.angles_deg.resize(contour.size());
  signal.gradient_deg.resize(contour.size());

  // 每个方向对应从 P[i] 指向 P[i+1] 的链码方向；最后一点自然回到第一点。
  cv::Point2f last_valid{1.0F, 0.0F};
  for (int i = 0; i < size; ++i) {
    const cv::Point2f difference =
      cv::Point2f(contour[static_cast<std::size_t>((i + 1) % size)] -
                  contour[static_cast<std::size_t>(i)]);
    const cv::Point2f direction = normalized(difference);
    if (direction.dot(direction) > 0.0F) last_valid = direction;
    signal.raw_directions[static_cast<std::size_t>(i)] = last_valid;
  }

  // 窗口随轮廓长度变化，同时受绝对上下限约束，兼顾不同目标尺度。
  const int requested_window = static_cast<int>(
    std::lround(params_.smoothing_window_ratio * static_cast<double>(size))) | 1;
  const int window = odd_clamped_window(
    requested_window, params_.min_smoothing_window, params_.max_smoothing_window, size);
  const std::vector<float> kernel = gaussian_kernel(window, params_.smoothing_sigma);
  const int radius = window / 2;

  // 对方向向量而非角度值做圆周卷积，天然避开 -180/180 度数值跳变。
  for (int i = 0; i < size; ++i) {
    cv::Point2f sum{};
    for (int offset = -radius; offset <= radius; ++offset) {
      sum += kernel[static_cast<std::size_t>(offset + radius)] *
             signal.raw_directions[static_cast<std::size_t>(circular_index(i + offset, size))];
    }
    signal.smoothed_directions[static_cast<std::size_t>(i)] = normalized(sum);
    const cv::Point2f & direction = signal.smoothed_directions[static_cast<std::size_t>(i)];
    signal.angles_deg[static_cast<std::size_t>(i)] =
      std::atan2(direction.y, direction.x) * kRadToDeg;
  }

  // 使用左右相邻平滑方向的圆周角作为中心差分梯度。
  for (int i = 0; i < size; ++i) {
    const cv::Point2f & previous =
      signal.smoothed_directions[static_cast<std::size_t>(circular_index(i - 1, size))];
    const cv::Point2f & next =
      signal.smoothed_directions[static_cast<std::size_t>(circular_index(i + 1, size))];
    signal.gradient_deg[static_cast<std::size_t>(i)] = signed_angle_deg(previous, next) * 0.5F;
  }
  return signal;
}

std::vector<BoundarySegment> FanSkeletonExtractor::extract_boundaries(
  const std::vector<cv::Point> & contour, const DirectionSignal & signal) const
{
  const int size = static_cast<int>(contour.size());
  // 低角度梯度意味着局部切向基本不变，可视为直线边界的一部分。
  std::vector<unsigned char> stable(contour.size(), 0U);
  for (int i = 0; i < size; ++i) {
    stable[static_cast<std::size_t>(i)] =
      std::abs(signal.gradient_deg[static_cast<std::size_t>(i)]) <=
          params_.max_tangent_gradient_deg
        ? 1U
        : 0U;
  }

  // 同时使用绝对阈值和相对阈值，使短缺口容限能随目标大小变化。
  const int max_gap = std::max(
    params_.max_straight_gap_samples,
    static_cast<int>(std::lround(params_.max_straight_gap_ratio * size)));
  close_short_unstable_gaps(
    stable, signal.smoothed_directions, max_gap, params_.max_straight_merge_angle_deg);

  // 对每个稳定区间拟合直线，并按有效长度与法向残差做最后过滤。
  const int min_samples = std::max(
    params_.min_segment_samples,
    static_cast<int>(std::ceil(params_.min_segment_samples_ratio * size)));
  std::vector<BoundarySegment> boundaries;
  for (const std::vector<int> & run : circular_true_runs(stable)) {
    if (static_cast<int>(run.size()) < min_samples) continue;
    BoundarySegment boundary = fit_boundary(contour, run, signal);
    const float max_rms = std::max(
      static_cast<float>(params_.max_segment_rms),
      static_cast<float>(params_.max_segment_rms_ratio) * boundary.length);
    if (boundary.length > kEpsilon && boundary.rms_error <= max_rms) {
      boundaries.push_back(boundary);
    }
  }
  return boundaries;
}

std::vector<BoundaryPair> FanSkeletonExtractor::match_boundaries(
  const std::vector<cv::Point> & contour,
  const std::vector<BoundarySegment> & boundaries) const
{
  if (boundaries.size() < 2U) return {};
  const int contour_size = static_cast<int>(contour.size());
  const cv::Rect bounding_box = cv::boundingRect(contour);
  const double bounding_diagonal = std::hypot(bounding_box.width, bounding_box.height);
  const double max_width = params_.max_strip_width_bbox_ratio * bounding_diagonal;

  // 第一轮枚举所有几何上可行的正反边界组合，并为每个组合计算质量分数。
  std::vector<PairCandidate> candidates;
  for (std::size_t i = 0; i + 1U < boundaries.size(); ++i) {
    for (std::size_t j = i + 1U; j < boundaries.size(); ++j) {
      const BoundarySegment & first = boundaries[i];
      const BoundarySegment & second = boundaries[j];
      // 同一灯条两侧在闭合轮廓上的遍历方向应接近相反。
      const float angle = undirected_angle_deg(first.direction, second.direction);
      const float opposite_error = std::abs(180.0F - angle);
      if (opposite_error > params_.opposite_angle_tolerance_deg) continue;

      // 取两个遍历方向中的较短轮廓间隔，排除相距过远但恰好平行的结构。
      const int arc_gap = std::min(
        forward_gap(first, second, contour_size),
        forward_gap(second, first, contour_size));
      if (arc_gap > params_.max_pair_arc_gap_ratio * contour_size) continue;

      // 两个反向单位向量作差得到统一轴向，旋转 90 度得到灯条法向。
      const cv::Point2f tangent = normalized(first.direction - second.direction);
      if (tangent.dot(tangent) <= 0.0F) continue;
      const cv::Point2f normal{-tangent.y, tangent.x};
      const cv::Point2f center_delta = second.center - first.center;
      const float width = std::abs(center_delta.dot(normal));
      const float max_boundary_length = std::max(first.length, second.length);
      const float min_boundary_length = std::min(first.length, second.length);
      if (width < params_.min_strip_width || width > max_width ||
          min_boundary_length / width < params_.min_pair_aspect_ratio) {
        continue;
      }
      const float longitudinal_offset = std::abs(center_delta.dot(tangent));
      if (longitudinal_offset >
          params_.max_pair_longitudinal_offset_ratio * max_boundary_length) {
        continue;
      }

      // 将两条边界投影到统一轴向，只使用公共区间恢复骨架，避免无依据外推。
      const auto interval = [&](const BoundarySegment & boundary) {
        float first_projection = boundary.start_point.dot(tangent);
        float second_projection = boundary.end_point.dot(tangent);
        if (first_projection > second_projection) std::swap(first_projection, second_projection);
        return std::pair<float, float>{first_projection, second_projection};
      };
      const auto first_interval = interval(first);
      const auto second_interval = interval(second);
      const float overlap_start = std::max(first_interval.first, second_interval.first);
      const float overlap_end = std::min(first_interval.second, second_interval.second);
      const float overlap = overlap_end - overlap_start;
      const float overlap_ratio = overlap / min_boundary_length;
      if (overlap <= 0.0F || overlap_ratio < params_.min_pair_overlap_ratio) continue;

      // 骨架法向坐标取两边界中值，轴向端点取上一步求得的重叠区间。
      const float normal_coordinate =
        0.5F * (first.center.dot(normal) + second.center.dot(normal));
      SkeletonLine skeleton;
      skeleton.start = tangent * overlap_start + normal * normal_coordinate;
      skeleton.end = tangent * overlap_end + normal * normal_coordinate;
      skeleton.center = (skeleton.start + skeleton.end) * 0.5F;
      skeleton.direction = tangent;
      skeleton.length = overlap;

      BoundaryPair pair;
      pair.first_boundary = static_cast<int>(i);
      pair.second_boundary = static_cast<int>(j);
      pair.skeleton = skeleton;
      pair.strip_width = width;
      pair.opposite_angle_error_deg = opposite_error;
      pair.overlap_ratio = overlap_ratio;
      // 分数综合方向、重叠、中心错位和拟合误差，数值越小越可信。
      pair.score =
        opposite_error / static_cast<float>(params_.opposite_angle_tolerance_deg) +
        (1.0F - std::min(overlap_ratio, 1.0F)) +
        longitudinal_offset / std::max(max_boundary_length, kEpsilon) +
        (first.rms_error + second.rms_error) /
          std::max(first.length + second.length, kEpsilon);
      candidates.push_back({static_cast<int>(i), static_cast<int>(j), pair});
    }
  }

  // 第二轮按质量从优到劣贪心选择，保证每条边界最多属于一根灯条。
  std::sort(
    candidates.begin(), candidates.end(),
    [](const PairCandidate & lhs, const PairCandidate & rhs) {
      if (lhs.pair.score != rhs.pair.score) return lhs.pair.score < rhs.pair.score;
      if (lhs.first != rhs.first) return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

  std::vector<unsigned char> used(boundaries.size(), 0U);
  std::vector<BoundaryPair> pairs;
  for (const PairCandidate & candidate : candidates) {
    if (used[static_cast<std::size_t>(candidate.first)] != 0U ||
        used[static_cast<std::size_t>(candidate.second)] != 0U) {
      continue;
    }
    used[static_cast<std::size_t>(candidate.first)] = 1U;
    used[static_cast<std::size_t>(candidate.second)] = 1U;
    pairs.push_back(candidate.pair);
  }
  return pairs;
}


// 将距离足够近的重复特征点合并，避免同一交点被多个线段组合重复输出。
void append_unique_point(
  std::vector<cv::Point2f> & points, const cv::Point2f & candidate, float merge_distance)
{
  const auto duplicate = std::find_if(
    points.begin(), points.end(), [&](const cv::Point2f & point) {
      return cv::norm(point - candidate) <= merge_distance;
    });
  if (duplicate == points.end()) points.push_back(candidate);
}

// 将 CHAIN_APPROX_SIMPLE 等稀疏轮廓按像素步长展开，满足方向信号的采样需求。
std::vector<cv::Point> densify_closed_contour(const std::vector<cv::Point> & contour)
{
  if (contour.size() < 2U) return contour;

  std::vector<cv::Point> dense;
  for (std::size_t index = 0; index < contour.size(); ++index) {
    const cv::Point start = contour[index];
    const cv::Point end = contour[(index + 1U) % contour.size()];
    const int steps = std::max(std::abs(end.x - start.x), std::abs(end.y - start.y));
    if (steps == 0) continue;

    for (int step = 0; step < steps; ++step) {
      const float ratio = static_cast<float>(step) / static_cast<float>(steps);
      const cv::Point point(
        cvRound(start.x + ratio * static_cast<float>(end.x - start.x)),
        cvRound(start.y + ratio * static_cast<float>(end.y - start.y)));
      if (dense.empty() || dense.back() != point) dense.push_back(point);
    }
  }
  return dense;
}

struct SkeletonCandidate
{
  SkeletonLine skeleton;
  float score = std::numeric_limits<float>::infinity();
};

// 骨架是无向轴，比较方向时将 180 度等价方向折叠到 [0, 90]。
float skeleton_axis_angle_deg(const SkeletonLine & first, const SkeletonLine & second)
{
  const float angle = undirected_angle_deg(first.direction, second.direction);
  return std::min(angle, 180.0F - angle);
}

FanSkeletonParams make_sensitive_params(const FanSkeletonParams & strict)
{
  FanSkeletonParams sensitive = strict;
  sensitive.max_tangent_gradient_deg = std::max(7.0, strict.max_tangent_gradient_deg);
  sensitive.min_segment_samples = std::min(4, strict.min_segment_samples);
  sensitive.min_smoothing_window = std::min(5, strict.min_smoothing_window);
  sensitive.smoothing_sigma = std::min(2.5, strict.smoothing_sigma);
  sensitive.max_segment_rms = std::max(3.5, strict.max_segment_rms);
  sensitive.min_strip_width = std::min(0.25, strict.min_strip_width);
  sensitive.opposite_angle_tolerance_deg =
    std::max(28.0, strict.opposite_angle_tolerance_deg);
  sensitive.min_pair_overlap_ratio = std::min(0.10, strict.min_pair_overlap_ratio);

  // 不放宽轮廓间隔、纵向错位和拐点合并角，避免不同叉臂之间错误配对。
  return sensitive;
}

std::vector<SkeletonCandidate> collect_candidates(
  const ContourSkeletonResult & result)
{
  std::vector<SkeletonCandidate> candidates;
  for (const BoundaryPair & pair : result.pairs) {
    candidates.push_back({pair.skeleton, pair.score});
  }
  return candidates;
}

bool same_skeleton(const SkeletonCandidate & first, const SkeletonCandidate & second)
{
  const float min_length = std::min(first.skeleton.length, second.skeleton.length);
  const float center_tolerance = std::max(5.0F, 0.15F * min_length);
  return skeleton_axis_angle_deg(first.skeleton, second.skeleton) <= 10.0F &&
         cv::norm(first.skeleton.center - second.skeleton.center) <= center_tolerance;
}

std::vector<SkeletonCandidate> merge_candidates_per_contour(
  const std::vector<ContourSkeletonResult> & strict_results,
  const std::vector<ContourSkeletonResult> & sensitive_results)
{
  constexpr std::size_t kMaxSkeletonsPerContour = 8U;
  std::vector<SkeletonCandidate> selected;
  const std::size_t contour_count = std::min(strict_results.size(), sensitive_results.size());
  for (std::size_t index = 0; index < contour_count; ++index) {
    std::vector<SkeletonCandidate> contour_selected =
      collect_candidates(strict_results[index]);
    std::sort(
      contour_selected.begin(), contour_selected.end(),
      [](const SkeletonCandidate & first, const SkeletonCandidate & second) {
        return first.score < second.score;
      });
    if (contour_selected.size() > kMaxSkeletonsPerContour) {
      contour_selected.resize(kMaxSkeletonsPerContour);
    }

    std::vector<SkeletonCandidate> sensitive =
      collect_candidates(sensitive_results[index]);
    std::sort(
      sensitive.begin(), sensitive.end(),
      [](const SkeletonCandidate & first, const SkeletonCandidate & second) {
        return first.score < second.score;
      });
    for (const SkeletonCandidate & candidate : sensitive) {
      if (contour_selected.size() >= kMaxSkeletonsPerContour) break;
      const bool duplicate = std::any_of(
        contour_selected.begin(), contour_selected.end(),
        [&](const SkeletonCandidate & existing) { return same_skeleton(existing, candidate); });
      if (!duplicate) contour_selected.push_back(candidate);
    }

    selected.insert(selected.end(), contour_selected.begin(), contour_selected.end());
  }
  return selected;
}





SkeletonFeaturePoints detect_and_show_skeleton(
  const cv::Mat & binary, const std::vector<std::vector<cv::Point>> & contours,
  const FanSkeletonParams & params, const std::string & window_name)
{
  // 在标记点旁绘制稳定的 ASCII 编号；OpenCV putText 默认不支持中文字形。
  auto draw_labeled_points = [](cv::Mat & image, const std::vector<cv::Point2f> & points,
                              const std::string & prefix, const cv::Scalar & color) {
    for (std::size_t index = 0; index < points.size(); ++index) {
      const cv::Point center(cvRound(points[index].x), cvRound(points[index].y));
      cv::circle(image, center, 5, color, cv::FILLED, cv::LINE_AA);
      cv::putText(
        image, prefix + std::to_string(index), center + cv::Point(5, -5),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
    }
  };

  // 计算两条骨架线段的交点，并允许交点在端点之外小范围延伸，以容忍拟合截断误差。
  auto skeleton_intersection = [](const SkeletonLine & first, const SkeletonLine & second,
                              cv::Point2f & intersection, float endpoint_tolerance = 4.0F) {
    const cv::Point2f first_vector = first.end - first.start;
    const cv::Point2f second_vector = second.end - second.start;
    const float denominator = first_vector.cross(second_vector);
    if (std::abs(denominator) < 1e-5F) return false;

    const cv::Point2f start_delta = second.start - first.start;
    const float first_ratio = start_delta.cross(second_vector) / denominator;
    const float second_ratio = start_delta.cross(first_vector) / denominator;
    const float first_margin = endpoint_tolerance / std::max(first.length, 1.0F);
    const float second_margin = endpoint_tolerance / std::max(second.length, 1.0F);
    if (first_ratio < -first_margin || first_ratio > 1.0F + first_margin ||
        second_ratio < -second_margin || second_ratio > 1.0F + second_margin) {
      return false;
    }

    intersection = first.start + first_ratio * first_vector;
    return true;
  };


  if (binary.empty() || binary.type() != CV_8UC1) {
    throw std::invalid_argument("binary image must be a non-empty CV_8UC1 matrix");
  }

  const FanSkeletonExtractor strict_extractor(params);
  const FanSkeletonExtractor sensitive_extractor(make_sensitive_params(params));
  std::vector<ContourSkeletonResult> strict_results;
  std::vector<ContourSkeletonResult> sensitive_results;
  strict_results.reserve(contours.size());
  sensitive_results.reserve(contours.size());
  for (std::size_t index = 0; index < contours.size(); ++index) {
    const std::vector<cv::Point> dense_contour = densify_closed_contour(contours[index]);
    if (dense_contour.size() < 3U) continue;
    strict_results.push_back(
      strict_extractor.analyze_contour(dense_contour, static_cast<int>(index)));
    sensitive_results.push_back(
      sensitive_extractor.analyze_contour(dense_contour, static_cast<int>(index)));
  }

  const std::vector<SkeletonCandidate> selected =
    merge_candidates_per_contour(strict_results, sensitive_results);

  cv::Mat visualization;
  cv::cvtColor(binary, visualization, cv::COLOR_GRAY2BGR);

  SkeletonFeaturePoints feature_points;
  for (const ContourSkeletonResult & result : strict_results) {
    const std::vector<std::vector<cv::Point>> contours{result.contour};
    cv::drawContours(visualization, contours, -1, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
  }
  for (const SkeletonCandidate & candidate : selected) {
    const SkeletonLine & skeleton = candidate.skeleton;
    feature_points.skeletons.push_back(skeleton);
    cv::line(
      visualization, skeleton.start, skeleton.end, cv::Scalar(0, 255, 255), 2,
      cv::LINE_AA);
    append_unique_point(feature_points.endpoints, skeleton.start, 3.0F);
    append_unique_point(feature_points.endpoints, skeleton.end, 3.0F);
  }

  // 枚举不同骨架组合；平行线无拐角，合法线段交点作为骨架拐角返回。
  for (std::size_t first = 0; first + 1U < feature_points.skeletons.size(); ++first) {
    for (std::size_t second = first + 1U; second < feature_points.skeletons.size(); ++second) {
      cv::Point2f intersection;
      if (skeleton_intersection(
            feature_points.skeletons[first], feature_points.skeletons[second], intersection)) {
        append_unique_point(feature_points.corners, intersection, 4.0F);
      }
    }
  }

  const cv::Scalar dark_red(0, 0, 255);
  draw_labeled_points(visualization, feature_points.endpoints, "E", dark_red);
  draw_labeled_points(visualization, feature_points.corners, "C", dark_red);

  feature_points.visualization = visualization;
  if (!window_name.empty()) cv::imshow(window_name, visualization);
  return feature_points;
}

GradientEndpointResult detect_and_show_gradient_endpoints(
  const cv::Mat & binary, const std::vector<std::vector<cv::Point>> & contours,
  const FanSkeletonParams & skeleton_params,
  const GradientEndpointParams & endpoint_params, const std::string & window_name)
{
  if (binary.empty() || binary.type() != CV_8UC1) {
    throw std::invalid_argument("binary image must be a non-empty CV_8UC1 matrix");
  }
  if (endpoint_params.min_gradient_deg < 0.0F ||
      endpoint_params.relative_gradient_ratio <= 0.0F ||
      endpoint_params.relative_gradient_ratio > 1.0F ||
      endpoint_params.convex_smoothing_radius < 0 ||
      endpoint_params.convex_smoothing_sigma < 0.0F || endpoint_params.local_max_radius < 1 ||
      endpoint_params.min_tip_arc_samples < 1 ||
      endpoint_params.min_tip_arc_limit_samples < endpoint_params.min_tip_arc_samples ||
      endpoint_params.max_tip_arc_ratio <= 0.0F ||
      endpoint_params.min_peak_merge_distance <= 0.0F ||
      endpoint_params.peak_merge_distance_bbox_ratio <= 0.0F ||
      endpoint_params.max_double_corner_valley_ratio < 0.0F ||
      endpoint_params.max_double_corner_valley_ratio >= 1.0F ||
      endpoint_params.endpoint_merge_distance <= 0.0F ||
      endpoint_params.min_unpaired_peak_ratio <= 0.0F ||
      endpoint_params.min_unpaired_peak_ratio > 1.0F ||
      endpoint_params.image_border_margin < 0 ||
      endpoint_params.max_endpoints_per_contour < 1) {
    throw std::invalid_argument("invalid gradient endpoint parameters");
  }

  struct GradientPeak
  {
    int index = -1;
    int run_offset = -1;
    cv::Point2f point{};
    float magnitude = 0.0F;
  };
  struct ConvexRegion
  {
    cv::Point2f endpoint{};
    float strength = 0.0F;
  };

  const auto near_border = [&](const cv::Point2f & point) {
    const float margin = static_cast<float>(endpoint_params.image_border_margin);
    return point.x <= margin || point.y <= margin ||
           point.x >= static_cast<float>(binary.cols - 1) - margin ||
           point.y >= static_cast<float>(binary.rows - 1) - margin;
  };

  const FanSkeletonExtractor extractor(skeleton_params);
  GradientEndpointResult output;
  cv::cvtColor(binary, output.visualization, cv::COLOR_GRAY2BGR);

  for (std::size_t contour_index = 0; contour_index < contours.size(); ++contour_index) {
    const std::vector<cv::Point> contour = densify_closed_contour(contours[contour_index]);
    if (contour.size() < 3U) continue;
    const ContourSkeletonResult analysis =
      extractor.analyze_contour(contour, static_cast<int>(contour_index));
    const std::vector<float> & gradients = analysis.signal.gradient_deg;
    if (gradients.empty()) continue;

    const std::vector<std::vector<cv::Point>> draw_contours{contour};
    cv::drawContours(output.visualization, draw_contours, -1, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);

    float signed_turn_sum = 0.0F;
    for (float gradient : gradients) {
      signed_turn_sum += gradient;
    }
    const float convex_sign = signed_turn_sum >= 0.0F ? 1.0F : -1.0F;
    const int contour_size = static_cast<int>(contour.size());

    // 对有符号梯度先低通再取凸响应，使尺寸很小、正负快速交替的毛刺互相抵消。
    const int smoothing_radius = std::min(
      endpoint_params.convex_smoothing_radius, std::max(0, (contour_size - 1) / 2));
    const std::vector<float> smoothing_kernel = gaussian_kernel(
      2 * smoothing_radius + 1, endpoint_params.convex_smoothing_sigma);
    std::vector<float> convex_response(contour.size(), 0.0F);
    float max_convex_magnitude = 0.0F;
    for (int index = 0; index < contour_size; ++index) {
      float filtered_gradient = 0.0F;
      for (int offset = -smoothing_radius; offset <= smoothing_radius; ++offset) {
        filtered_gradient +=
          smoothing_kernel[static_cast<std::size_t>(offset + smoothing_radius)] *
          gradients[static_cast<std::size_t>(circular_index(index + offset, contour_size))];
      }
      const float response = std::max(0.0F, filtered_gradient * convex_sign);
      convex_response[static_cast<std::size_t>(index)] = response;
      max_convex_magnitude = std::max(max_convex_magnitude, response);
    }
    if (max_convex_magnitude <= 0.0F) continue;

    const float threshold = std::max(
      endpoint_params.min_gradient_deg,
      endpoint_params.relative_gradient_ratio * max_convex_magnitude);

    std::vector<unsigned char> convex_mask(contour.size(), 0U);
    for (int index = 0; index < contour_size; ++index) {
      const cv::Point2f point(contour[static_cast<std::size_t>(index)]);
      if (!near_border(point) && convex_response[static_cast<std::size_t>(index)] >= threshold) {
        convex_mask[static_cast<std::size_t>(index)] = 1U;
      }
    }

    const cv::Rect bounding_box = cv::boundingRect(contour);
    const float bounding_diagonal = static_cast<float>(
      std::hypot(bounding_box.width, bounding_box.height));
    const float max_merge_distance = std::max(
      endpoint_params.min_peak_merge_distance,
      endpoint_params.peak_merge_distance_bbox_ratio * bounding_diagonal);
    const int max_region_gap = std::max(
      endpoint_params.min_tip_arc_limit_samples,
      static_cast<int>(std::lround(endpoint_params.max_tip_arc_ratio * contour_size)));

    // 将轮廓上相邻且空间接近的凸响应合为一个区域。平头端帽的两个直角峰
    // 会跨过中间短直线合并，而相距较远的不同叉臂仍保持为独立区域。
    const auto first_convex = std::find(convex_mask.begin(), convex_mask.end(), 1U);
    if (first_convex == convex_mask.end()) continue;
    const int seed = static_cast<int>(std::distance(convex_mask.begin(), first_convex));
    int visited = 0;
    int scan_index = circular_index(seed + 1, contour_size);
    while (visited < contour_size) {
      if (convex_mask[static_cast<std::size_t>(scan_index)] != 0U) {
        scan_index = circular_index(scan_index + 1, contour_size);
        ++visited;
        continue;
      }

      const int gap_start = scan_index;
      int gap_length = 0;
      while (visited + gap_length < contour_size &&
             convex_mask[static_cast<std::size_t>(scan_index)] == 0U) {
        scan_index = circular_index(scan_index + 1, contour_size);
        ++gap_length;
      }
      visited += gap_length;

      const int left = circular_index(gap_start - 1, contour_size);
      const int right = scan_index;
      bool gap_touches_border = false;
      for (int offset = 0; offset < gap_length && !gap_touches_border; ++offset) {
        gap_touches_border = near_border(
          cv::Point2f(contour[static_cast<std::size_t>(
            circular_index(gap_start + offset, contour_size))]));
      }
      if (gap_length <= max_region_gap && !gap_touches_border &&
          convex_mask[static_cast<std::size_t>(right)] != 0U &&
          cv::norm(cv::Point2f(contour[static_cast<std::size_t>(left)]) -
                   cv::Point2f(contour[static_cast<std::size_t>(right)])) <=
            max_merge_distance) {
        for (int offset = 0; offset < gap_length; ++offset) {
          convex_mask[static_cast<std::size_t>(
            circular_index(gap_start + offset, contour_size))] = 1U;
        }
      }
    }

    std::vector<ConvexRegion> regions;
    for (const std::vector<int> & run : circular_true_runs(convex_mask)) {
      if (run.empty()) continue;

      std::vector<GradientPeak> peaks;
      for (std::size_t run_offset = 0; run_offset < run.size(); ++run_offset) {
        const int index = run[run_offset];
        const float magnitude = convex_response[static_cast<std::size_t>(index)];
        bool local_maximum = true;
        bool has_lower_neighbor = false;
        for (int offset = -endpoint_params.local_max_radius;
             offset <= endpoint_params.local_max_radius; ++offset) {
          if (offset == 0) continue;
          const int neighbor = circular_index(index + offset, contour_size);
          const float neighbor_magnitude = convex_response[static_cast<std::size_t>(neighbor)];
          if (neighbor_magnitude > magnitude) {
            local_maximum = false;
            break;
          }
          has_lower_neighbor = has_lower_neighbor || neighbor_magnitude < magnitude;
        }
        if (local_maximum && has_lower_neighbor) {
          peaks.push_back(
            {index, static_cast<int>(run_offset), cv::Point2f(contour[static_cast<std::size_t>(index)]),
             magnitude});
        }
      }

      // 极平的响应平台可能没有严格主峰，退化为区域内响应最大的位置。
      if (peaks.empty()) {
        const auto strongest = std::max_element(
          run.begin(), run.end(), [&](int first, int second) {
            return convex_response[static_cast<std::size_t>(first)] <
                   convex_response[static_cast<std::size_t>(second)];
          });
        const int run_offset = static_cast<int>(std::distance(run.begin(), strongest));
        peaks.push_back(
          {*strongest, run_offset, cv::Point2f(contour[static_cast<std::size_t>(*strongest)]),
           convex_response[static_cast<std::size_t>(*strongest)]});
      }
      std::sort(
        peaks.begin(), peaks.end(),
        [](const GradientPeak & first, const GradientPeak & second) {
          return first.magnitude > second.magnitude;
        });

      const GradientPeak & strongest = peaks.front();
      cv::Point2f endpoint = strongest.point;
      float region_strength = strongest.magnitude;
      for (std::size_t second_index = 1; second_index < peaks.size(); ++second_index) {
        const GradientPeak & second = peaks[second_index];
        const int run_gap = std::abs(strongest.run_offset - second.run_offset);
        if (run_gap < endpoint_params.min_tip_arc_samples ||
            second.magnitude < endpoint_params.min_unpaired_peak_ratio * strongest.magnitude ||
            cv::norm(strongest.point - second.point) > max_merge_distance) {
          continue;
        }

        const int begin = std::min(strongest.run_offset, second.run_offset);
        const int end = std::max(strongest.run_offset, second.run_offset);
        float valley = std::numeric_limits<float>::infinity();
        for (int offset = begin + 1; offset < end; ++offset) {
          valley = std::min(
            valley, convex_response[static_cast<std::size_t>(
                      run[static_cast<std::size_t>(offset)])]);
        }
        if (valley <= endpoint_params.max_double_corner_valley_ratio * second.magnitude) {
          endpoint = 0.5F * (strongest.point + second.point);
          region_strength += second.magnitude;
          break;
        }
      }

      if (!near_border(endpoint)) regions.push_back({endpoint, region_strength});
    }

    // 每个凸起区域最多贡献一个端点；数量超限时优先保留响应最强的区域。
    std::sort(
      regions.begin(), regions.end(),
      [](const ConvexRegion & first, const ConvexRegion & second) {
        return first.strength > second.strength;
      });
    const std::size_t retained_count = std::min(
      regions.size(), static_cast<std::size_t>(endpoint_params.max_endpoints_per_contour));
    for (std::size_t index = 0; index < retained_count; ++index) {
      append_unique_point(
        output.endpoints, regions[index].endpoint, endpoint_params.endpoint_merge_distance);
    }
  }

  const cv::Scalar red(0, 0, 255);
  for (std::size_t index = 0; index < output.endpoints.size(); ++index) {
    const cv::Point center(
      cvRound(output.endpoints[index].x), cvRound(output.endpoints[index].y));
    cv::circle(output.visualization, center, 5, red, cv::FILLED, cv::LINE_AA);
    cv::putText(
      output.visualization, "E" + std::to_string(index), center + cv::Point(5, -5),
      cv::FONT_HERSHEY_SIMPLEX, 0.45, red, 1, cv::LINE_AA);
  }

  if (!window_name.empty()) cv::imshow(window_name, output.visualization);
  return output;
}

}  // namespace auto_buff::kami_rune
