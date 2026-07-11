#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
const int LOSE_MAX = 5;  // 丢失的阙值
namespace auto_buff
{
class Buff_Detector
{
public:
  Buff_Detector(const std::string & config);

  std::optional<BuffObservation> detect_24(
    cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp);

  std::optional<BuffObservation> detect_24(cv::Mat & bgr_img);

  std::optional<BuffObservation> detect(
    cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp);

  std::optional<BuffObservation> detect(cv::Mat & bgr_img);

  std::optional<BuffObservation> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

  int gate_failure_count() const { return gate_failure_count_; }

  int confirmed_switch_count() const { return confirmed_switch_count_; }

private:
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  void handle_lose();

  struct CenterEstimate
  {
    cv::Point2f point;
    RuneCenterSource source;
  };

  std::optional<BuffObservation> detect_impl(
    cv::Mat & bgr_img, bool single_candidate,
    std::chrono::steady_clock::time_point timestamp);

  std::vector<BuffObservation> build_candidates(
    const std::vector<YOLO11_BUFF::Object> & results, const cv::Point2f & r_center) const;

  std::optional<CenterEstimate> select_r_center(
    const std::vector<YOLO11_BUFF::Object> & results,
    std::chrono::steady_clock::time_point timestamp);

  std::optional<BuffObservation> select_locked_candidate(
    const std::vector<BuffObservation> & candidates,
    std::chrono::steady_clock::time_point timestamp);

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  double lastlen_;

  float keypoint_threshold_ = 0.3f;
  int center_lost_max_ = 6;

  double pair_angle_gate_rad_ = 15.0 / 57.3;
  double pair_ratio_min_ = 0.30;
  double pair_ratio_max_ = 0.70;
  double pair_ratio_center_ = 0.51;
  double track_gate_min_rad_ = 12.0 / 57.3;
  double track_gate_max_rad_ = 25.0 / 57.3;
  double blind_timeout_s_ = 0.080;
  double track_reset_timeout_s_ = 0.500;
  int switch_confirm_frames_ = 5;

  bool has_last_r_center_ = false;
  cv::Point2f last_r_center_{0.0f, 0.0f};
  int center_lost_count_ = 0;
  std::chrono::steady_clock::time_point last_r_center_time_{};

  bool has_locked_target_ = false;
  double last_locked_angle_ = 0.0;
  double angular_velocity_ = 0.0;
  int next_track_id_ = 1;
  int locked_track_id_ = -1;
  int lost_locked_count_ = 0;
  bool has_pending_switch_ = false;
  double pending_switch_angle_ = 0.0;
  std::chrono::steady_clock::time_point pending_switch_time_{};
  int switch_confirm_count_ = 0;
  std::chrono::steady_clock::time_point last_locked_time_{};
  std::chrono::steady_clock::time_point last_seen_time_{};
  bool gate_episode_active_ = false;
  int gate_failure_count_ = 0;
  int confirmed_switch_count_ = 0;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
