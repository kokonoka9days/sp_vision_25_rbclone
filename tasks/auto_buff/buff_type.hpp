#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
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
inline double RUNE_RADIUS_M = 0.700;
inline int SMALL_BUFF_DIRECTION = 0;  // 0: auto, 1/-1: force small buff prediction direction
inline constexpr double RUNE_SLOT_ANGLE = 2.0 * CV_PI / 5.0;

enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

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

  explicit FanBlade() = default;

  // explicit FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(
    const std::vector<cv::Point2f> & target_kpt, const std::vector<cv::Point2f> & fan_kpt,
    cv::Point2f target_center, cv::Point2f fan_center, FanBlade_type t, float confidence,
    int slot_id);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;  // 按target开始顺时针

  int light_num;
  int target_slot_id = -1;
  double target_angle = 0.0;
  int positive_roll_image_direction = 0;

  Eigen::Vector3d xyz_in_world;  // 单位：m
  Eigen::Vector3d ypr_in_world;  // 单位：rad
  Eigen::Vector3d ypd_in_world;  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world;  // 单位：m
  Eigen::Vector3d blade_ypd_in_world;  // 球坐标系, 单位: m

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP
