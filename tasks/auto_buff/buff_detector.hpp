#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "buff_config.hpp"
#include "buff_track_bank.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
const int LOSE_MAX = 5;  // 丢失的阙值
namespace auto_buff
{
class Buff_Detector
{
public:
  /** @brief 从配置文件初始化能量机关检测器 @param config YAML 配置路径 */
  Buff_Detector(const std::string & config);
  /** @brief 使用已解析配置初始化检测器 @param config_path YAML 配置路径 @param config 能量机关配置 */
  Buff_Detector(const std::string & config_path, BuffConfig config);

  /** @brief 检测并更新全部能量机关轨迹 @param bgr_img BGR 图像 @param mode 小符或大符模式 @param timestamp 采集时间戳 @return 已排序的轨迹观测 */
  std::vector<BuffObservation> detect_tracks(
    cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp);

  /** @brief 使用当前时间检测全部轨迹 @param bgr_img BGR 图像 @param mode 小符或大符模式 @return 已排序的轨迹观测 */
  std::vector<BuffObservation> detect_tracks(cv::Mat & bgr_img, BuffMode mode);

  /** @brief 使用 2024 单候选策略检测 @param bgr_img BGR 图像 @param mode 模式 @param timestamp 采集时间戳 @return 主目标观测 */
  std::optional<BuffObservation> detect_24(
    cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp);

  /** @brief 使用当前模式和指定时间执行单候选检测 @param bgr_img BGR 图像 @param timestamp 采集时间戳 @return 主目标观测 */
  std::optional<BuffObservation> detect_24(
    cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp);

  /** @brief 使用当前模式与当前时间执行单候选检测 @param bgr_img BGR 图像 @return 主目标观测 */
  std::optional<BuffObservation> detect_24(cv::Mat & bgr_img);

  /** @brief 检测指定模式的主目标 @param bgr_img BGR 图像 @param mode 模式 @param timestamp 采集时间戳 @return 主目标观测 */
  std::optional<BuffObservation> detect(
    cv::Mat & bgr_img, BuffMode mode, std::chrono::steady_clock::time_point timestamp);

  /** @brief 使用当前模式检测主目标 @param bgr_img BGR 图像 @param timestamp 采集时间戳 @return 主目标观测 */
  std::optional<BuffObservation> detect(
    cv::Mat & bgr_img, std::chrono::steady_clock::time_point timestamp);

  /** @brief 使用当前模式和当前时间检测主目标 @param bgr_img BGR 图像 @return 主目标观测 */
  std::optional<BuffObservation> detect(cv::Mat & bgr_img);

  /** @brief 使用给定方向向量执行调试检测 @param bgr_img BGR 图像 @param v 参考方向向量 @return 调试观测 */
  std::optional<BuffObservation> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

  /** @brief 获取门限失败次数 @return 当前实现固定返回 0 */
  int gate_failure_count() const { return 0; }

  /** @brief 获取轨迹主目标确认切换次数 @return 切换次数 */
  int confirmed_switch_count() const { return track_bank_.confirmed_switch_count(); }
  /** @brief 获取时序门限拒绝次数 @return 拒绝次数 */
  int temporal_reject_count() const { return track_bank_.temporal_reject_count(); }

private:
  const BuffConfig config_;
  /** @brief 对输入图像执行颜色和形态学预处理 @param bgr_img BGR 图像 @param dilated_img 输出膨胀二值图 */
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  /** @brief 根据扇叶估计 R 标中心 @param fanblades 扇叶列表 @param bgr_img 调试图像 @return R 标像素中心 */
  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  /** @brief 更新目标丢失计数和状态 */
  void handle_lose();

  struct CenterEstimate
  {
    cv::Point2f point;
    RuneCenterSource source;
  };

  /** @brief 检测实现 @param bgr_img BGR 图像 @param single_candidate 是否仅使用最高置信候选 @param mode 模式 @param timestamp 时间戳 @return 主目标观测 */
  std::optional<BuffObservation> detect_impl(
    cv::Mat & bgr_img, bool single_candidate, BuffMode mode,
    std::chrono::steady_clock::time_point timestamp);

  /** @brief 多轨迹检测实现 @param bgr_img BGR 图像 @param single_candidate 是否仅使用最高置信候选 @param mode 模式 @param timestamp 时间戳 @return 轨迹观测列表 */
  std::vector<BuffObservation> detect_tracks_impl(
    cv::Mat & bgr_img, bool single_candidate, BuffMode mode,
    std::chrono::steady_clock::time_point timestamp);

  /** @brief 在模式变化时重置时序状态 @param mode 新模式 */
  void reset_for_mode(BuffMode mode);

  /** @brief 将网络结果配对为候选观测 @param results 网络检测对象 @param r_center R 标中心 @return 候选观测列表 */
  std::vector<BuffObservation> build_candidates(
    const std::vector<YOLO11_BUFF::Object> & results, const cv::Point2f & r_center) const;

  /** @brief 从检测结果或运动预测选择 R 标中心 @param results 网络检测对象 @param timestamp 时间戳 @return 中心及其来源 */
  std::optional<CenterEstimate> select_r_center(
    const std::vector<YOLO11_BUFF::Object> & results,
    std::chrono::steady_clock::time_point timestamp);

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  double lastlen_;

  float keypoint_threshold_ = 0.3f;
  float hard_keypoint_threshold_ = 0.15f;
  double temporal_residual_gate_px_ = 10.0;
  int center_lost_max_ = 6;
  double center_innovation_gate_px_ = 45.0;
  int center_recovery_hits_ = 2;
  double center_retention_s_ = 0.400;

  double pair_angle_gate_rad_ = 15.0 / 57.3;
  double pair_ratio_min_ = 0.30;
  double pair_ratio_max_ = 0.70;
  double pair_ratio_center_ = 0.51;
  double track_gate_min_rad_ = 12.0 / 57.3;
  double track_gate_max_rad_ = 25.0 / 57.3;
  double blind_timeout_s_ = 0.080;
  double track_reset_timeout_s_ = 0.500;
  int switch_confirm_frames_ = 5;
  int same_slot_confirm_frames_ = 3;
  int adjacent_switch_confirm_frames_ = 8;
  double adjacent_switch_delay_s_ = 0.180;
  double slot_tolerance_rad_ = 12.0 / 57.3;
  double switch_pair_angle_gate_rad_ = 10.0 / 57.3;
  double switch_pair_ratio_min_ = 0.38;
  double switch_pair_ratio_max_ = 0.64;

  bool has_last_r_center_ = false;
  cv::Point2f last_r_center_{0.0f, 0.0f};
  int center_lost_count_ = 0;
  std::chrono::steady_clock::time_point last_r_center_time_{};
  std::chrono::steady_clock::time_point last_r_center_seen_time_{};
  cv::Point2f r_center_velocity_{0.0f, 0.0f};
  bool has_pending_r_center_ = false;
  cv::Point2f pending_r_center_{0.0f, 0.0f};
  int pending_r_center_hits_ = 0;

  BuffTrackBank track_bank_;
  BuffMode current_mode_ = BuffMode::SMALL;
  bool has_current_mode_ = false;

};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
