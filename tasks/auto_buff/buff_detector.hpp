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

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  void handle_lose();

  std::optional<PowerRune> detect_impl(cv::Mat & bgr_img, bool single_candidate);

  std::vector<FanBlade> build_candidates(
    const std::vector<YOLO11_BUFF::Object> & results, const cv::Point2f & r_center) const;

  std::optional<cv::Point2f> select_r_center(
    const std::vector<YOLO11_BUFF::Object> & results);

  std::optional<FanBlade> select_locked_candidate(const std::vector<FanBlade> & candidates);

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  double lastlen_;
  std::optional<PowerRune> last_powerrune_ = std::nullopt;

  float keypoint_threshold_ = 0.3f;
  double locked_gate_rad_ = 25.0 / 57.3;
  double switch_gate_rad_ = 10.0 / 57.3;
  int switch_confirm_frames_ = 3;
  int locked_lost_max_ = 6;
  int center_lost_max_ = 6;

  bool has_last_r_center_ = false;
  cv::Point2f last_r_center_{0.0f, 0.0f};
  int center_lost_count_ = 0;

  bool has_locked_target_ = false;
  double last_locked_angle_ = 0.0;
  int locked_slot_id_ = -1;
  int lost_locked_count_ = 0;
  int pending_switch_slot_id_ = -1;
  int switch_confirm_count_ = 0;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
