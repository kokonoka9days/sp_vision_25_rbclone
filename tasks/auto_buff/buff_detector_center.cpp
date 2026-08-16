#include "buff_detector.hpp"

#include <algorithm>

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

/** @brief 检查检测对象是否具有四个可信关键点 @param object 检测对象 @param threshold 置信度阈值 @return 有效时返回 true */
bool valid_keypoints(const YOLO11_BUFF::Object & object, float threshold)
{
  if (object.kpt.size() != 4) return false;
  if (object.kpt_conf.empty()) return true;
  return std::all_of(object.kpt_conf.begin(), object.kpt_conf.end(), [&](float confidence) {
    return confidence >= threshold;
  });
}

/** @brief 计算两个稳定时钟时间点之差 @param now 当前时间 @param before 先前时间 @return 时间差，单位 s */
double seconds_between(
  std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point before)
{
  return std::chrono::duration<double>(now - before).count();
}
}  // namespace

std::optional<Buff_Detector::CenterEstimate> Buff_Detector::select_r_center(
  const std::vector<YOLO11_BUFF::Object> & results,
  std::chrono::steady_clock::time_point timestamp)
{
  const YOLO11_BUFF::Object * best_center = nullptr;
  for (const auto & result : results) {
    if (result.label != RUNE_CENTER || !valid_keypoints(result, keypoint_threshold_)) continue;
    if (best_center == nullptr || result.prob > best_center->prob) best_center = &result;
  }

  if (!has_last_r_center_) {
    if (best_center == nullptr) return std::nullopt;
    last_r_center_ = mean_points(best_center->kpt);
    has_last_r_center_ = true;
    center_lost_count_ = 0;
    last_r_center_time_ = timestamp;
    last_r_center_seen_time_ = timestamp;
    return CenterEstimate{last_r_center_, RuneCenterSource::DETECTED};
  }

  const double dt = std::max(0.0, seconds_between(timestamp, last_r_center_time_));
  const cv::Point2f predicted = last_r_center_ + r_center_velocity_ * static_cast<float>(dt);
  if (best_center != nullptr) {
    const cv::Point2f observed = mean_points(best_center->kpt);
    const double residual = cv::norm(observed - predicted);
    if (residual <= center_innovation_gate_px_) {
      if (dt > 1e-4) {
        cv::Point2f measured_velocity = (observed - last_r_center_) * static_cast<float>(1.0 / dt);
        const double speed = cv::norm(measured_velocity);
        if (speed > 1500.0) measured_velocity *= static_cast<float>(1500.0 / speed);
        r_center_velocity_ = 0.65f * r_center_velocity_ + 0.35f * measured_velocity;
      }
      last_r_center_ = observed;
      last_r_center_time_ = timestamp;
      last_r_center_seen_time_ = timestamp;
      center_lost_count_ = 0;
      has_pending_r_center_ = false;
      pending_r_center_hits_ = 0;
      return CenterEstimate{last_r_center_, RuneCenterSource::DETECTED};
    }

    if (
      has_pending_r_center_ &&
      cv::norm(observed - pending_r_center_) <= center_innovation_gate_px_) {
      pending_r_center_hits_++;
      pending_r_center_ = observed;
    } else {
      has_pending_r_center_ = true;
      pending_r_center_ = observed;
      pending_r_center_hits_ = 1;
    }
    if (pending_r_center_hits_ >= center_recovery_hits_) {
      last_r_center_ = pending_r_center_;
      r_center_velocity_ = {0.0f, 0.0f};
      last_r_center_time_ = timestamp;
      last_r_center_seen_time_ = timestamp;
      has_pending_r_center_ = false;
      pending_r_center_hits_ = 0;
      return CenterEstimate{last_r_center_, RuneCenterSource::DETECTED};
    }
  }

  if (seconds_between(timestamp, last_r_center_seen_time_) <= center_retention_s_) {
    last_r_center_ = predicted;
    last_r_center_time_ = timestamp;
    center_lost_count_++;
    return CenterEstimate{last_r_center_, RuneCenterSource::PREDICTED};
  }

  return std::nullopt;
}
}  // namespace auto_buff
