#include "buff_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
/** @brief 计算像素点集均值 @param points 点集 @return 平均点 */
cv::Point2f mean_points(const std::vector<cv::Point2f> & points)
{
  cv::Point2f sum{0.0f, 0.0f};
  for (const auto & point : points) sum += point;
  if (!points.empty()) sum *= 1.0f / static_cast<float>(points.size());
  return sum;
}

/** @brief 计算二维点积 @param a 向量一 @param b 向量二 @return 点积 */
float dot2d(const cv::Point2f & a, const cv::Point2f & b) { return a.x * b.x + a.y * b.y; }

/** @brief 计算二维叉积标量 @param a 向量一 @param b 向量二 @return 叉积 */
float cross2d(const cv::Point2f & a, const cv::Point2f & b) { return a.x * b.y - a.y * b.x; }

/** @brief 归一化向量或使用后备方向 @param v 输入向量 @param fallback 长度过小时的后备向量 @return 单位向量 */
cv::Point2f normalize_or(const cv::Point2f & v, const cv::Point2f & fallback)
{
  const float n = cv::norm(v);
  if (n < 1e-3f) return fallback;
  return v * (1.0f / n);
}

/** @brief 检查检测对象是否具有四个可信关键点 @param object 检测对象 @param threshold 置信度阈值 @return 有效时返回 true */
bool valid_keypoints(const YOLO11_BUFF::Object & object, float threshold)
{
  if (object.kpt.size() != 4) return false;
  if (object.kpt_conf.empty()) return true;
  return std::all_of(object.kpt_conf.begin(), object.kpt_conf.end(), [&](float conf) {
    return conf >= threshold;
  });
}

/** @brief 获取检测对象关键点的最低置信度 @param object 检测对象 @return 最低置信度 */
float minimum_keypoint_confidence(const YOLO11_BUFF::Object & object)
{
  if (object.kpt_conf.empty()) return 1.0f;
  return *std::min_element(object.kpt_conf.begin(), object.kpt_conf.end());
}

/** @brief 将目标四点匹配到上右下左语义顺序 @param points 输入四点 @param center 中心 @param axis_y 局部纵轴 @return 排序点；输入无效时为空 */
std::optional<std::vector<cv::Point2f>> order_target_points(
  const std::vector<cv::Point2f> & points, const cv::Point2f & center,
  const cv::Point2f & axis_y)
{
  if (points.size() != 4) return std::nullopt;
  const cv::Point2f axis_x{axis_y.y, -axis_y.x};
  const std::array<cv::Point2f, 4> expected{-axis_y, axis_x, axis_y, -axis_x};
  std::array<int, 4> permutation{0, 1, 2, 3};
  std::array<int, 4> index = permutation;
  double best_cost = std::numeric_limits<double>::max();
  do {
    double cost = 0.0;
    for (int role = 0; role < 4; ++role) {
      const cv::Point2f direction =
        normalize_or(points[permutation[role]] - center, expected[role]);
      cost += 1.0 - dot2d(direction, expected[role]);
    }
    if (cost < best_cost) {
      best_cost = cost;
      index = permutation;
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  return std::vector<cv::Point2f>{
    points[index[0]], points[index[1]], points[index[2]], points[index[3]]};
}

/** @brief 将矩形四点排列为左上、右上、右下、左下 @param points 输入点 @param center 中心 @param axis_y 局部纵轴 @return 排序后的点 */
std::vector<cv::Point2f> order_rect_points(
  const std::vector<cv::Point2f> & points, const cv::Point2f & center,
  const cv::Point2f & axis_y)
{
  std::vector<int> indexes(points.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::sort(indexes.begin(), indexes.end(), [&](int lhs, int rhs) {
    return dot2d(points[lhs] - center, axis_y) < dot2d(points[rhs] - center, axis_y);
  });

  std::array<int, 2> top{indexes[0], indexes[1]};
  std::array<int, 2> bottom{indexes[2], indexes[3]};
  auto left_first = [&](int lhs, int rhs) {
    return cross2d(axis_y, points[lhs] - center) > cross2d(axis_y, points[rhs] - center);
  };
  std::sort(top.begin(), top.end(), left_first);
  std::sort(bottom.begin(), bottom.end(), left_first);

  return {points[top[0]], points[top[1]], points[bottom[1]], points[bottom[0]]};
}

/** @brief 评估有序四边形的形状质量 @param points 四个顶点 @return 0 到 1 的质量分数 */
double quad_quality_score(const std::vector<cv::Point2f> & points)
{
  if (points.size() != 4) return 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      if (cv::norm(points[i] - points[j]) < 3.0) return 0.0;
    }
  }
  std::vector<cv::Point2f> hull;
  cv::convexHull(points, hull);
  const double area = std::abs(cv::contourArea(hull));
  if (hull.size() != 4 || area < 20.0) return 0.0;

  std::array<double, 4> sides{};
  for (size_t i = 0; i < points.size(); ++i) {
    sides[i] = cv::norm(points[(i + 1) % points.size()] - points[i]);
  }
  const double opposite_a = std::min(sides[0], sides[2]) / std::max(sides[0], sides[2]);
  const double opposite_b = std::min(sides[1], sides[3]) / std::max(sides[1], sides[3]);
  const double compactness = std::clamp(area / 100.0, 0.0, 1.0);
  return std::min({opposite_a, opposite_b, 0.5 + 0.5 * compactness});
}

/** @brief 检查有序四边形是否达到最低质量 @param points 四个顶点 @return 有效时返回 true */
bool valid_ordered_quad(const std::vector<cv::Point2f> & points)
{
  return quad_quality_score(points) >= 0.10;
}

/** @brief 绕旧中心旋转点集并平移到新中心 @param points 输入点集 @param old_center 旧中心 @param new_center 新中心 @param angle 旋转角 @return 变换点集 */
std::vector<cv::Point2f> rotate_points(
  const std::vector<cv::Point2f> & points, const cv::Point2f & old_center,
  const cv::Point2f & new_center, double angle)
{
  const float c = static_cast<float>(std::cos(angle));
  const float s = static_cast<float>(std::sin(angle));
  std::vector<cv::Point2f> rotated;
  rotated.reserve(points.size());
  for (const auto & point : points) {
    const cv::Point2f relative = point - old_center;
    rotated.emplace_back(
      new_center + cv::Point2f(c * relative.x - s * relative.y, s * relative.x + c * relative.y));
  }
  return rotated;
}

/** @brief 计算四边形平均半径 @param points 四个顶点 @return 平均半径 */
double quad_scale(const std::vector<cv::Point2f> & points)
{
  if (points.size() != 4) return 0.0;
  const cv::Point2f center = mean_points(points);
  double radius = 0.0;
  for (const auto & point : points) radius += cv::norm(point - center);
  return radius / 4.0;
}

/** @brief 计算点绕中心的极角 @param point 查询点 @param center 中心 @return 极角，单位 rad */
double angle_around(const cv::Point2f & point, const cv::Point2f & center)
{
  return std::atan2(point.y - center.y, point.x - center.x);
}

/** @brief 获取观测完整度排序值 @param type 观测类型 @return 越小越优先 */
int observation_rank(BuffObservationType type)
{
  switch (type) {
    case BuffObservationType::FULL:
      return 0;
    case BuffObservationType::TARGET_ONLY:
      return 1;
    case BuffObservationType::FAN_ONLY:
      return 2;
  }
  return 3;
}

}  // namespace

Buff_Detector::Buff_Detector(const std::string & config)
: Buff_Detector(config, load_buff_config(config))
{
}

Buff_Detector::Buff_Detector(const std::string & config, BuffConfig buff_config)
: config_(std::move(buff_config)), MODE_(config), status_(LOSE), lose_(0)
{
  auto yaml = YAML::LoadFile(config);
  if (yaml["buff_keypoint_threshold"]) keypoint_threshold_ = yaml["buff_keypoint_threshold"].as<float>();
  if (yaml["buff_keypoint_hard_threshold"]) {
    hard_keypoint_threshold_ = yaml["buff_keypoint_hard_threshold"].as<float>();
  }
  if (yaml["buff_keypoint_temporal_gate_px"]) {
    temporal_residual_gate_px_ = yaml["buff_keypoint_temporal_gate_px"].as<double>();
  }
  if (yaml["buff_center_innovation_gate_px"]) {
    center_innovation_gate_px_ = yaml["buff_center_innovation_gate_px"].as<double>();
  }
  if (yaml["buff_center_recovery_hits"]) {
    center_recovery_hits_ = yaml["buff_center_recovery_hits"].as<int>();
  }
  if (yaml["buff_center_lost_max"]) center_lost_max_ = yaml["buff_center_lost_max"].as<int>();
  if (yaml["buff_pair_angle_gate_deg"]) {
    pair_angle_gate_rad_ = yaml["buff_pair_angle_gate_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_pair_ratio_min"]) pair_ratio_min_ = yaml["buff_pair_ratio_min"].as<double>();
  if (yaml["buff_pair_ratio_max"]) pair_ratio_max_ = yaml["buff_pair_ratio_max"].as<double>();
  if (yaml["buff_pair_ratio_center"]) {
    pair_ratio_center_ = yaml["buff_pair_ratio_center"].as<double>();
  }
  if (yaml["buff_track_gate_min_deg"]) {
    track_gate_min_rad_ = yaml["buff_track_gate_min_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_track_gate_max_deg"]) {
    track_gate_max_rad_ = yaml["buff_track_gate_max_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_blind_timeout_s"]) blind_timeout_s_ = yaml["buff_blind_timeout_s"].as<double>();
  if (yaml["buff_track_reset_timeout_s"]) {
    track_reset_timeout_s_ = yaml["buff_track_reset_timeout_s"].as<double>();
  }
  if (yaml["buff_switch_confirm_frames"]) {
    switch_confirm_frames_ = yaml["buff_switch_confirm_frames"].as<int>();
    adjacent_switch_confirm_frames_ = switch_confirm_frames_;
  }
  if (yaml["buff_same_slot_confirm_frames"]) {
    same_slot_confirm_frames_ = yaml["buff_same_slot_confirm_frames"].as<int>();
  }
  if (yaml["buff_adjacent_switch_confirm_frames"]) {
    adjacent_switch_confirm_frames_ = yaml["buff_adjacent_switch_confirm_frames"].as<int>();
  }
  if (yaml["buff_adjacent_switch_delay_s"]) {
    adjacent_switch_delay_s_ = yaml["buff_adjacent_switch_delay_s"].as<double>();
  }
  if (yaml["buff_slot_tolerance_deg"]) {
    slot_tolerance_rad_ = yaml["buff_slot_tolerance_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_switch_pair_angle_gate_deg"]) {
    switch_pair_angle_gate_rad_ =
      yaml["buff_switch_pair_angle_gate_deg"].as<double>() / 57.3;
  }
  if (yaml["buff_switch_pair_ratio_min"]) {
    switch_pair_ratio_min_ = yaml["buff_switch_pair_ratio_min"].as<double>();
  }
  if (yaml["buff_switch_pair_ratio_max"]) {
    switch_pair_ratio_max_ = yaml["buff_switch_pair_ratio_max"].as<double>();
  }
  blind_timeout_s_ = config_.blind_timeout_s;

  BuffTrackBank::Config track_config;
  track_config.confirm_hits =
    yaml["buff_track_confirm_hits"] ? yaml["buff_track_confirm_hits"].as<int>() : 2;
  track_config.recovery_hits =
    yaml["buff_track_recovery_hits"] ? yaml["buff_track_recovery_hits"].as<int>() : 2;
  track_config.association_gate_rad = track_gate_max_rad_;
  track_config.point_residual_gate_px = temporal_residual_gate_px_;
  track_config.control_blind_timeout_s =
    yaml["buff_control_blind_timeout_s"]
      ? yaml["buff_control_blind_timeout_s"].as<double>()
      : config_.blind_timeout_s;
  track_config.retention_timeout_s =
    yaml["buff_track_retention_s"]
      ? yaml["buff_track_retention_s"].as<double>()
      : config_.track_retention_s;
  track_config.spawn_keypoint_threshold = keypoint_threshold_;
  center_retention_s_ = track_config.retention_timeout_s;
  blind_timeout_s_ = track_config.control_blind_timeout_s;
  track_bank_.configure(track_config);
}

void Buff_Detector::handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img)
{
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 100, 255, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), 1);
}

cv::Point2f Buff_Detector::get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img)
{
  if (fanblades.empty()) {
    tools::logger()->debug("[Buff_Detector] 无法计算r_center!");
    return {0, 0};
  }

  cv::Point2f r_center_t = {0, 0};
  int valid_count = 0;
  for (auto & fanblade : fanblades) {
    if (fanblade.fan_center != cv::Point2f(0, 0)) {
      const auto axis = normalize_or(fanblade.fan_center - fanblade.center, {0.0f, 1.0f});
      r_center_t += fanblade.center +
                    axis * static_cast<float>(
                             cv::norm(fanblade.fan_center - fanblade.center) * 2.0);
      valid_count++;
    }
  }
  if (valid_count == 0) return fanblades.front().center;
  r_center_t *= 1.0f / static_cast<float>(valid_count);

  cv::Mat dilated_img;
  handle_img(bgr_img, dilated_img);
  double radius = cv::norm(fanblades[0].center - r_center_t) * 0.25;
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U);
  circle(mask, r_center_t, radius, cv::Scalar(255), -1);
  bitwise_and(dilated_img, mask, dilated_img);

  std::vector<std::vector<cv::Point>> contours;
  auto r_center = r_center_t;
  cv::findContours(dilated_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  double ratio_1 = INF;
  for (auto & it : contours) {
    auto rotated_rect = cv::minAreaRect(it);
    double ratio = rotated_rect.size.height > rotated_rect.size.width
                     ? rotated_rect.size.height / rotated_rect.size.width
                     : rotated_rect.size.width / rotated_rect.size.height;
    ratio += cv::norm(rotated_rect.center - r_center_t) / std::max(radius / 3.0, 1.0);
    if (ratio < ratio_1) {
      ratio_1 = ratio;
      r_center = rotated_rect.center;
    }
  }
  return r_center;
}

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
  } else {
    status_ = TEM_LOSE;
  }
}

std::vector<BuffObservation> Buff_Detector::build_candidates(
  const std::vector<YOLO11_BUFF::Object> & results, const cv::Point2f & r_center) const
{
  struct Detection
  {
    const YOLO11_BUFF::Object * object;
    cv::Point2f center;
  };
  struct PairEdge
  {
    int target;
    int fan;
    double angle_error;
    double ratio;
    double score;
  };

  std::vector<Detection> targets;
  std::vector<Detection> fans;
  for (const auto & result : results) {
    if (!valid_keypoints(result, hard_keypoint_threshold_)) continue;
    if (result.label == INACTIVE_TARGET) targets.push_back({&result, mean_points(result.kpt)});
    if (result.label == INACTIVE_FAN) fans.push_back({&result, mean_points(result.kpt)});
  }

  std::vector<PairEdge> edges;
  for (int ti = 0; ti < static_cast<int>(targets.size()); ++ti) {
    const auto & target = targets[ti];
    const double target_radius = cv::norm(target.center - r_center);
    if (target_radius < 3.0) continue;
    for (int fi = 0; fi < static_cast<int>(fans.size()); ++fi) {
      const auto & fan = fans[fi];
      const double fan_radius = cv::norm(fan.center - r_center);
      const cv::Point2f r_to_fan = fan.center - r_center;
      const cv::Point2f fan_to_target = target.center - fan.center;
      const double fan_target_distance = cv::norm(fan_to_target);
      if (fan_radius < 3.0 || fan_target_distance < 3.0 || target_radius <= fan_radius) continue;

      const cv::Point2f u = normalize_or(r_to_fan, {0.0f, 1.0f});
      const cv::Point2f v = normalize_or(fan_to_target, {0.0f, 1.0f});
      const double direction_dot = std::clamp(static_cast<double>(dot2d(u, v)), -1.0, 1.0);
      if (direction_dot <= 0.0) continue;
      const double angle_error = std::acos(direction_dot);
      const double ratio = fan_target_distance / target_radius;
      if (
        angle_error > pair_angle_gate_rad_ || ratio < pair_ratio_min_ ||
        ratio > pair_ratio_max_) {
        continue;
      }

      const double confidence = 0.5 * (target.object->prob + fan.object->prob);
      const double angle_term = angle_error / std::max(pair_angle_gate_rad_, 1e-6);
      const double ratio_term = (ratio - pair_ratio_center_) / 0.20;
      const double score = angle_term * angle_term + ratio_term * ratio_term + (1.0 - confidence);
      edges.push_back({ti, fi, angle_error, ratio, score});
    }
  }

  std::vector<int> current_match(targets.size(), -1);
  std::vector<int> best_match(targets.size(), -1);
  std::vector<bool> used_fans(fans.size(), false);
  int best_count = -1;
  double best_score = std::numeric_limits<double>::max();
  std::function<void(int, int, double)> assign = [&](int ti, int count, double score) {
    if (ti == static_cast<int>(targets.size())) {
      if (count > best_count || (count == best_count && score < best_score)) {
        best_count = count;
        best_score = score;
        best_match = current_match;
      }
      return;
    }

    current_match[ti] = -1;
    assign(ti + 1, count, score);
    for (int ei = 0; ei < static_cast<int>(edges.size()); ++ei) {
      const auto & edge = edges[ei];
      if (edge.target != ti || used_fans[edge.fan]) continue;
      current_match[ti] = ei;
      used_fans[edge.fan] = true;
      assign(ti + 1, count + 1, score + edge.score);
      used_fans[edge.fan] = false;
      current_match[ti] = -1;
    }
  };
  assign(0, 0, 0.0);

  std::vector<BuffObservation> candidates;
  std::vector<bool> matched_targets(targets.size(), false);
  std::vector<bool> matched_fans(fans.size(), false);
  for (int ti = 0; ti < static_cast<int>(best_match.size()); ++ti) {
    if (best_match[ti] < 0) continue;
    const auto & edge = edges[best_match[ti]];
    const auto & target = targets[ti];
    const auto & fan = fans[edge.fan];
    const cv::Point2f axis_to_center = normalize_or(r_center - target.center, {0.0f, 1.0f});
    const auto ordered_target = order_target_points(target.object->kpt, target.center, axis_to_center);
    const auto ordered_fan = order_rect_points(fan.object->kpt, fan.center, axis_to_center);
    if (!ordered_target.has_value() || !valid_ordered_quad(*ordered_target) ||
        !valid_ordered_quad(ordered_fan)) {
      continue;
    }

    BuffObservation observation;
    observation.type = BuffObservationType::FULL;
    observation.target_points = *ordered_target;
    observation.fan_points = ordered_fan;
    observation.raw_target_points = observation.target_points;
    observation.raw_fan_points = observation.fan_points;
    observation.target_center = target.center;
    observation.fan_center = fan.center;
    observation.target_center_observed = true;
    observation.fan_center_observed = true;
    observation.angle = angle_around(target.center, r_center);
    observation.pair_angle_error = edge.angle_error;
    observation.pair_distance_ratio = edge.ratio;
    observation.confidence = 0.5f * (target.object->prob + fan.object->prob);
    observation.min_keypoint_confidence = std::min(
      minimum_keypoint_confidence(*target.object), minimum_keypoint_confidence(*fan.object));
    observation.quad_quality =
      std::min(quad_quality_score(*ordered_target), quad_quality_score(ordered_fan));
    observation.association_cost = edge.score;
    if (observation.min_keypoint_confidence < keypoint_threshold_) {
      const double confidence_span = std::max(
        static_cast<double>(keypoint_threshold_ - hard_keypoint_threshold_), 1e-3);
      observation.keypoint_noise_scale =
        1.0 + 2.0 * (keypoint_threshold_ - observation.min_keypoint_confidence) / confidence_span;
    }
    candidates.emplace_back(std::move(observation));
    matched_targets[ti] = true;
    matched_fans[edge.fan] = true;
  }

  for (int ti = 0; ti < static_cast<int>(targets.size()); ++ti) {
    if (matched_targets[ti]) continue;
    const auto & target = targets[ti];
    const cv::Point2f axis_to_center = normalize_or(r_center - target.center, {0.0f, 1.0f});
    const auto ordered_target = order_target_points(target.object->kpt, target.center, axis_to_center);
    if (!ordered_target.has_value() || !valid_ordered_quad(*ordered_target)) continue;

    BuffObservation observation;
    observation.type = BuffObservationType::TARGET_ONLY;
    observation.target_points = *ordered_target;
    observation.raw_target_points = observation.target_points;
    observation.target_center = target.center;
    observation.target_center_observed = true;
    observation.fan_center = r_center + 0.49f * (target.center - r_center);
    observation.angle = angle_around(target.center, r_center);
    observation.confidence = target.object->prob;
    observation.min_keypoint_confidence = minimum_keypoint_confidence(*target.object);
    observation.quad_quality = quad_quality_score(*ordered_target);
    observation.association_cost = 1.0 - observation.confidence;
    if (observation.min_keypoint_confidence < keypoint_threshold_) {
      const double confidence_span = std::max(
        static_cast<double>(keypoint_threshold_ - hard_keypoint_threshold_), 1e-3);
      observation.keypoint_noise_scale =
        1.0 + 2.0 * (keypoint_threshold_ - observation.min_keypoint_confidence) / confidence_span;
    }
    candidates.emplace_back(std::move(observation));
  }

  for (int fi = 0; fi < static_cast<int>(fans.size()); ++fi) {
    if (matched_fans[fi]) continue;
    const auto & fan = fans[fi];
    const cv::Point2f axis_to_center = normalize_or(r_center - fan.center, {0.0f, 1.0f});
    const auto ordered_fan = order_rect_points(fan.object->kpt, fan.center, axis_to_center);
    if (!valid_ordered_quad(ordered_fan)) continue;

    BuffObservation observation;
    observation.type = BuffObservationType::FAN_ONLY;
    observation.fan_points = ordered_fan;
    observation.raw_fan_points = observation.fan_points;
    observation.fan_center = fan.center;
    observation.fan_center_observed = true;
    observation.target_center = r_center + (1.0f / 0.49f) * (fan.center - r_center);
    observation.angle = angle_around(fan.center, r_center);
    observation.confidence = fan.object->prob;
    observation.min_keypoint_confidence = minimum_keypoint_confidence(*fan.object);
    observation.quad_quality = quad_quality_score(ordered_fan);
    observation.association_cost = 1.0 - observation.confidence;
    if (observation.min_keypoint_confidence < keypoint_threshold_) {
      const double confidence_span = std::max(
        static_cast<double>(keypoint_threshold_ - hard_keypoint_threshold_), 1e-3);
      observation.keypoint_noise_scale =
        1.0 + 2.0 * (keypoint_threshold_ - observation.min_keypoint_confidence) / confidence_span;
    }
    candidates.emplace_back(std::move(observation));
  }

  std::sort(candidates.begin(), candidates.end(), [](const BuffObservation & a, const BuffObservation & b) {
    if (observation_rank(a.type) != observation_rank(b.type)) {
      return observation_rank(a.type) < observation_rank(b.type);
    }
    return a.confidence > b.confidence;
  });
  return candidates;
}

void Buff_Detector::reset_for_mode(BuffMode mode)
{
  if (has_current_mode_ && current_mode_ == mode) return;
  current_mode_ = mode;
  has_current_mode_ = true;
  track_bank_.reset();
  has_last_r_center_ = false;
  r_center_velocity_ = {0.0f, 0.0f};
  has_pending_r_center_ = false;
  pending_r_center_hits_ = 0;
  status_ = LOSE;
  lose_ = 0;
}

std::vector<BuffObservation> Buff_Detector::detect_tracks_impl(
  cv::Mat & bgr_img, bool single_candidate, BuffMode mode,
  std::chrono::steady_clock::time_point timestamp)
{
  reset_for_mode(mode);
  std::vector<YOLO11_BUFF::Object> results = single_candidate ? MODE_.get_onecandidatebox(bgr_img)
                                                              : MODE_.get_multicandidateboxes(bgr_img);
  if (results.empty()) {
    track_bank_.update({}, timestamp, mode);
    handle_lose();
    return {};
  }

  auto r_center = select_r_center(results, timestamp);
  if (!r_center.has_value()) {
    track_bank_.update({}, timestamp, mode);
    handle_lose();
    return {};
  }

  auto candidates = build_candidates(results, r_center->point);
  for (auto & candidate : candidates) {
    candidate.r_center = r_center->point;
    candidate.center_source = r_center->source;
    candidate.timestamp = timestamp;
  }
  auto tracked = track_bank_.update(candidates, timestamp, mode);
  if (tracked.empty()) {
    handle_lose();
    return {};
  }

  tools::draw_point(bgr_img, r_center->point, {0, 0, 255}, 3);
  for (const auto & observation : tracked) {
    if (observation.target_center_observed) {
      tools::draw_point(
        bgr_img, observation.target_center,
        observation.primary ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 255, 0), 3);
    }
    if (observation.fan_center_observed) {
      tools::draw_point(
        bgr_img, observation.fan_center,
        observation.primary ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255), 3);
    }
  }

  status_ = TRACK;
  lose_ = 0;
  return tracked;
}

std::optional<BuffObservation> Buff_Detector::detect_impl(
  cv::Mat & bgr_img, bool single_candidate, BuffMode mode,
  std::chrono::steady_clock::time_point timestamp)
{
  auto tracks = detect_tracks_impl(bgr_img, single_candidate, mode, timestamp);
  if (tracks.empty()) return std::nullopt;
  return tracks.front();
}

std::vector<BuffObservation> Buff_Detector::detect_tracks(
  cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp)
{
  return detect_tracks_impl(bgr_img, false, mode, timestamp);
}

std::vector<BuffObservation> Buff_Detector::detect_tracks(cv::Mat & bgr_img, BuffMode mode)
{
  return detect_tracks(bgr_img, mode, std::chrono::steady_clock::now());
}

std::optional<BuffObservation> Buff_Detector::detect_24(
  cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp)
{
  return detect_impl(bgr_img, false, mode, timestamp);
}

std::optional<BuffObservation> Buff_Detector::detect_24(
  cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp)
{
  return detect_impl(bgr_img, false, BuffMode::SMALL, timestamp);
}

std::optional<BuffObservation> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  return detect_24(bgr_img, std::chrono::steady_clock::now());
}

std::optional<BuffObservation> Buff_Detector::detect(
  cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp)
{
  return detect_impl(bgr_img, false, mode, timestamp);
}

std::optional<BuffObservation> Buff_Detector::detect(
  cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp)
{
  return detect_impl(bgr_img, false, BuffMode::SMALL, timestamp);
}

std::optional<BuffObservation> Buff_Detector::detect(cv::Mat & bgr_img)
{
  return detect(bgr_img, std::chrono::steady_clock::now());
}

std::optional<BuffObservation> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f)
{
  return detect(bgr_img, std::chrono::steady_clock::now());
}

}  // namespace auto_buff
