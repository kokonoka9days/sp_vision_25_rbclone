#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <chrono>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
inline constexpr double RUNE_SLOT_ANGLE = 2.0 * CV_PI / 5.0;

enum PowerRune_type { SMALL, BIG };
enum class BuffMode { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

enum class BuffObservationType { FULL, TARGET_ONLY, FAN_ONLY };
enum class RuneCenterSource { DETECTED, PREDICTED };
enum class BuffPoseQuality { FULL_8_POINT, PARTIAL_5_POINT, PARTIAL_4_POINT };
enum class BuffTrackStatus { TENTATIVE, CONFIRMED, COASTING };
enum class TargetReadiness { LOST, TRACKING, PREDICTING };

struct BuffObservation
{
  BuffObservationType type = BuffObservationType::FULL;
  RuneCenterSource center_source = RuneCenterSource::DETECTED;
  cv::Point2f r_center{0.0f, 0.0f};
  std::vector<cv::Point2f> target_points;
  std::vector<cv::Point2f> fan_points;
  std::vector<cv::Point2f> raw_target_points;
  std::vector<cv::Point2f> raw_fan_points;
  cv::Point2f target_center{0.0f, 0.0f};
  cv::Point2f fan_center{0.0f, 0.0f};
  bool target_center_observed = false;
  bool fan_center_observed = false;
  double angle = 0.0;
  double pair_angle_error = 0.0;
  double pair_distance_ratio = 0.0;
  double prediction_error = 0.0;
  double keypoint_temporal_residual = 0.0;
  double quad_quality = 1.0;
  double association_cost = 0.0;
  double slot_residual = 0.0;
  double keypoint_noise_scale = 1.0;
  float confidence = 0.0f;
  float min_keypoint_confidence = 1.0f;
  int slot_offset = 0;
  int track_id = -1;
  BuffTrackStatus track_status = BuffTrackStatus::TENTATIVE;
  bool primary = false;
  std::chrono::steady_clock::time_point timestamp{};

  /** @brief 查询是否包含完整目标四点 @return 完整时返回 true */
  bool has_target() const { return target_points.size() == 4; }
  /** @brief 查询是否包含完整扇叶四点 @return 完整时返回 true */
  bool has_fan() const { return fan_points.size() == 4; }
};

class FanBlade
{
public:
  cv::Point2f center;               // 扇页中心
  cv::Point2f fan_center;           // 扇叶外侧矩形中心
  std::vector<cv::Point2f> points;  // inactive_target四点: top, right, bottom, left
  std::vector<cv::Point2f> fan_points;  // inactive_fan四点: top-left, top-right, bottom-right, bottom-left
  double angle, width, height;
  int slot_id = -1;
  float confidence = 0.0f;
  FanBlade_type type;  // 类型

  /** @brief 构造空扇叶 */
  explicit FanBlade() = default;

  // explicit FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  /** @brief 由单组关键点构造扇叶 @param kpt 关键点 @param keypoints_center 关键点中心 @param t 扇叶类型 */
  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  /** @brief 由目标和扇叶关键点构造完整扇叶 @param target_kpt 目标四点 @param fan_kpt 扇叶四点 @param target_center 目标中心 @param fan_center 扇叶中心 @param t 扇叶类型 @param confidence 置信度 @param slot_id 槽位编号 */
  explicit FanBlade(
    const std::vector<cv::Point2f> & target_kpt, const std::vector<cv::Point2f> & fan_kpt,
    cv::Point2f target_center, cv::Point2f fan_center, FanBlade_type t, float confidence,
    int slot_id);

  /** @brief 仅使用类型构造扇叶 @param t 扇叶类型 */
  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;  // 按target开始顺时针

  int light_num;
  int target_slot_id = -1;
  int track_id = -1;
  int slot_offset = 0;
  bool primary = false;
  BuffTrackStatus track_status = BuffTrackStatus::TENTATIVE;
  double target_angle = 0.0;
  double slot_residual = 0.0;
  RuneCenterSource center_source = RuneCenterSource::DETECTED;
  BuffObservationType observation_type = BuffObservationType::FULL;
  BuffPoseQuality pose_quality = BuffPoseQuality::FULL_8_POINT;
  double measurement_noise_scale = 1.0;
  double reprojection_error = 0.0;
  double prediction_error = 0.0;

  Eigen::Vector3d xyz_in_world;  // 单位：m
  Eigen::Vector3d ypd_in_world;  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world;  // 单位：m
  Eigen::Vector3d blade_ypd_in_world;  // 球坐标系, 单位: m
  Eigen::Vector3d plane_normal_in_world{1.0, 0.0, 0.0};

  /** @brief 由扇叶列表构造能量机关 @param ts 扇叶列表 @param r_center R 标中心 @param last_powerrune 上一帧结果 */
  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  /** @brief 由二维观测构造待求解能量机关 @param observation 二维观测 */
  explicit PowerRune(const BuffObservation & observation);
  /** @brief 构造空能量机关 */
  explicit PowerRune() = default;

  /** @brief 获取目标扇叶 @return 目标扇叶引用 */
  FanBlade & target() { return fanblades[0]; };

  /** @brief 查询三维位姿是否无有效解 @return 无解时返回 true */
  bool is_unsolve() const { return unsolvable_; }

private:
  bool unsolvable_ = false;

  /** @brief 将二维向量转换为 [0, 2pi] 极角 @param v 二维向量 @return 极角，单位 rad */
  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP
