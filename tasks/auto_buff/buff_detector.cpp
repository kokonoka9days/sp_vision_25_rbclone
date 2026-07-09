#include "buff_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
cv::Point2f mean_points(const std::vector<cv::Point2f> & points)
{
  cv::Point2f sum{0.0f, 0.0f};
  for (const auto & point : points) sum += point;
  if (!points.empty()) sum *= 1.0f / static_cast<float>(points.size());
  return sum;
}

float dot2d(const cv::Point2f & a, const cv::Point2f & b) { return a.x * b.x + a.y * b.y; }

float cross2d(const cv::Point2f & a, const cv::Point2f & b) { return a.x * b.y - a.y * b.x; }

cv::Point2f normalize_or(const cv::Point2f & v, const cv::Point2f & fallback)
{
  const float n = cv::norm(v);
  if (n < 1e-3f) return fallback;
  return v * (1.0f / n);
}

bool valid_keypoints(const YOLO11_BUFF::Object & object, float threshold)
{
  if (object.kpt.size() != 4) return false;
  if (object.kpt_conf.empty()) return true;
  return std::all_of(object.kpt_conf.begin(), object.kpt_conf.end(), [&](float conf) {
    return conf >= threshold;
  });
}

std::vector<cv::Point2f> order_target_points(
  const std::vector<cv::Point2f> & points, const cv::Point2f & center,
  const cv::Point2f & axis_y)
{
  std::array<int, 4> index{};
  index.fill(0);
  float min_dot = std::numeric_limits<float>::max();
  float max_dot = -std::numeric_limits<float>::max();
  float min_cross = std::numeric_limits<float>::max();
  float max_cross = -std::numeric_limits<float>::max();

  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    const auto rel = points[i] - center;
    const float d = dot2d(rel, axis_y);
    const float c = cross2d(axis_y, rel);
    if (d < min_dot) {
      min_dot = d;
      index[0] = i;
    }
    if (d > max_dot) {
      max_dot = d;
      index[2] = i;
    }
    if (c < min_cross) {
      min_cross = c;
      index[1] = i;
    }
    if (c > max_cross) {
      max_cross = c;
      index[3] = i;
    }
  }

  return {points[index[0]], points[index[1]], points[index[2]], points[index[3]]};
}

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

int slot_id_from_angle(double angle)
{
  int slot = static_cast<int>(std::floor((angle + RUNE_SLOT_ANGLE * 0.5) / RUNE_SLOT_ANGLE));
  slot %= 5;
  if (slot < 0) slot += 5;
  return slot;
}
}  // namespace

Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0), MODE_(config)
{
  auto yaml = YAML::LoadFile(config);
  if (yaml["buff_keypoint_threshold"]) keypoint_threshold_ = yaml["buff_keypoint_threshold"].as<float>();
  if (yaml["buff_locked_gate_deg"]) locked_gate_rad_ = yaml["buff_locked_gate_deg"].as<double>() / 57.3;
  if (yaml["buff_switch_gate_deg"]) switch_gate_rad_ = yaml["buff_switch_gate_deg"].as<double>() / 57.3;
  if (yaml["buff_switch_confirm_frames"]) {
    switch_confirm_frames_ = yaml["buff_switch_confirm_frames"].as<int>();
  }
  if (yaml["buff_locked_lost_max"]) locked_lost_max_ = yaml["buff_locked_lost_max"].as<int>();
  if (yaml["buff_center_lost_max"]) center_lost_max_ = yaml["buff_center_lost_max"].as<int>();
  if (yaml["buff_rune_radius_m"]) RUNE_RADIUS_M = yaml["buff_rune_radius_m"].as<double>();
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
  if (has_locked_target_) lost_locked_count_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
  } else {
    status_ = TEM_LOSE;
  }
}

std::optional<cv::Point2f> Buff_Detector::select_r_center(
  const std::vector<YOLO11_BUFF::Object> & results)
{
  const YOLO11_BUFF::Object * best_center = nullptr;
  for (const auto & result : results) {
    if (result.label != RUNE_CENTER || !valid_keypoints(result, keypoint_threshold_)) continue;
    if (best_center == nullptr || result.prob > best_center->prob) best_center = &result;
  }

  if (best_center != nullptr) {
    last_r_center_ = mean_points(best_center->kpt);
    has_last_r_center_ = true;
    center_lost_count_ = 0;
    return last_r_center_;
  }

  if (has_last_r_center_ && center_lost_count_ < center_lost_max_) {
    center_lost_count_++;
    return last_r_center_;
  }

  return std::nullopt;
}

std::vector<FanBlade> Buff_Detector::build_candidates(
  const std::vector<YOLO11_BUFF::Object> & results, const cv::Point2f & r_center) const
{
  std::vector<const YOLO11_BUFF::Object *> targets;
  std::vector<const YOLO11_BUFF::Object *> fans;
  for (const auto & result : results) {
    if (!valid_keypoints(result, keypoint_threshold_)) continue;
    if (result.label == INACTIVE_TARGET) targets.push_back(&result);
    if (result.label == INACTIVE_FAN) fans.push_back(&result);
  }

  std::vector<FanBlade> candidates;
  for (const auto * target : targets) {
    const cv::Point2f target_center = mean_points(target->kpt);
    const cv::Point2f axis_to_center = normalize_or(r_center - target_center, {0.0f, 1.0f});
    const float center_distance =
      std::max(static_cast<float>(cv::norm(r_center - target_center)), 1.0f);

    const YOLO11_BUFF::Object * best_fan = nullptr;
    float best_score = std::numeric_limits<float>::max();
    cv::Point2f best_fan_center{0.0f, 0.0f};
    for (const auto * fan : fans) {
      const cv::Point2f fan_center = mean_points(fan->kpt);
      const cv::Point2f target_to_fan = fan_center - target_center;
      const float fan_distance = cv::norm(target_to_fan);
      if (fan_distance < 1.0f) continue;

      const cv::Point2f fan_axis = target_to_fan * (1.0f / fan_distance);
      const float direction_penalty = dot2d(axis_to_center, fan_axis) < 0.0f ? 2.0f : 0.0f;
      const float angle_penalty = std::abs(cross2d(axis_to_center, fan_axis));
      const float ratio_penalty = std::abs(fan_distance / center_distance - 0.5f);
      const float score = angle_penalty + 0.4f * ratio_penalty + direction_penalty;
      if (score < best_score) {
        best_score = score;
        best_fan = fan;
        best_fan_center = fan_center;
      }
    }

    if (best_fan == nullptr) continue;

    const auto ordered_target = order_target_points(target->kpt, target_center, axis_to_center);
    const auto ordered_fan = order_rect_points(best_fan->kpt, best_fan_center, axis_to_center);
    double angle = std::atan2(target_center.y - r_center.y, target_center.x - r_center.x);
    if (angle < 0.0) angle += CV_2PI;
    const int slot_id = slot_id_from_angle(angle);
    FanBlade candidate(
      ordered_target, ordered_fan, target_center, best_fan_center, _light,
      0.5f * (target->prob + best_fan->prob), slot_id);
    candidate.angle = angle;
    candidates.emplace_back(candidate);
  }

  std::sort(candidates.begin(), candidates.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.confidence > b.confidence;
  });
  return candidates;
}

std::optional<FanBlade> Buff_Detector::select_locked_candidate(
  const std::vector<FanBlade> & candidates)
{
  if (candidates.empty()) return std::nullopt;

  auto best_by_confidence = std::max_element(
    candidates.begin(), candidates.end(),
    [](const FanBlade & a, const FanBlade & b) { return a.confidence < b.confidence; });

  if (!has_locked_target_) {
    has_locked_target_ = true;
    lost_locked_count_ = 0;
    switch_confirm_count_ = 0;
    pending_switch_slot_id_ = -1;
    last_locked_angle_ = best_by_confidence->angle;
    locked_slot_id_ = best_by_confidence->slot_id;
    return *best_by_confidence;
  }

  auto best_by_angle = std::min_element(
    candidates.begin(), candidates.end(), [&](const FanBlade & a, const FanBlade & b) {
      return std::abs(tools::limit_rad(a.angle - last_locked_angle_)) <
             std::abs(tools::limit_rad(b.angle - last_locked_angle_));
    });
  const double best_error = std::abs(tools::limit_rad(best_by_angle->angle - last_locked_angle_));
  if (best_error <= locked_gate_rad_) {
    lost_locked_count_ = 0;
    switch_confirm_count_ = 0;
    pending_switch_slot_id_ = -1;
    last_locked_angle_ = best_by_angle->angle;
    locked_slot_id_ = best_by_angle->slot_id;
    return *best_by_angle;
  }

  lost_locked_count_++;
  if (lost_locked_count_ <= locked_lost_max_) {
    tools::logger()->debug(
      "[Buff_Detector] Locked target gated, err {:.1f} deg, blind predict",
      best_error * 57.3);
    return std::nullopt;
  }

  const double switch_error =
    std::abs(tools::limit_rad(best_by_confidence->angle - last_locked_angle_));
  if (switch_error < switch_gate_rad_) return std::nullopt;

  if (pending_switch_slot_id_ != best_by_confidence->slot_id) {
    pending_switch_slot_id_ = best_by_confidence->slot_id;
    switch_confirm_count_ = 1;
  } else {
    switch_confirm_count_++;
  }

  if (switch_confirm_count_ < switch_confirm_frames_) return std::nullopt;

  tools::logger()->debug("[Buff_Detector] Relock rune slot {}", best_by_confidence->slot_id);
  has_locked_target_ = true;
  lost_locked_count_ = 0;
  switch_confirm_count_ = 0;
  pending_switch_slot_id_ = -1;
  last_locked_angle_ = best_by_confidence->angle;
  locked_slot_id_ = best_by_confidence->slot_id;
  return *best_by_confidence;
}

std::optional<PowerRune> Buff_Detector::detect_impl(cv::Mat & bgr_img, bool single_candidate)
{
  std::vector<YOLO11_BUFF::Object> results = single_candidate ? MODE_.get_onecandidatebox(bgr_img)
                                                              : MODE_.get_multicandidateboxes(bgr_img);
  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  auto r_center = select_r_center(results);
  if (!r_center.has_value()) {
    handle_lose();
    return std::nullopt;
  }

  auto candidates = build_candidates(results, *r_center);
  auto locked_candidate = select_locked_candidate(candidates);
  if (!locked_candidate.has_value()) {
    handle_lose();
    return std::nullopt;
  }

  std::vector<FanBlade> fanblades{*locked_candidate};
  PowerRune powerrune(fanblades, *r_center, last_powerrune_);
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  tools::draw_point(bgr_img, *r_center, {0, 0, 255}, 3);
  tools::draw_point(bgr_img, locked_candidate->center, {255, 0, 255}, 3);

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  return detect_impl(bgr_img, false);
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  return detect_impl(bgr_img, false);
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f)
{
  return detect_impl(bgr_img, false);
}

}  // namespace auto_buff
