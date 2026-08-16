#ifndef AUTO_BUFF__TRACK_BANK_HPP
#define AUTO_BUFF__TRACK_BANK_HPP

#include <chrono>
#include <optional>
#include <vector>

#include "buff_type.hpp"

namespace auto_buff
{
class BuffTrackBank
{
public:
  struct Config
  {
    int confirm_hits = 2;
    int recovery_hits = 2;
    double association_gate_rad = 25.0 / 57.3;
    double point_residual_gate_px = 10.0;
    double control_blind_timeout_s = 0.100;
    double retention_timeout_s = 0.400;
    float spawn_keypoint_threshold = 0.30f;
  };

  /** @brief 使用默认参数构造轨迹库 */
  BuffTrackBank();
  /** @brief 使用指定参数构造轨迹库 @param config 关联与保留配置 */
  explicit BuffTrackBank(const Config & config);

  /** @brief 更新轨迹库配置 @param config 新配置 */
  void configure(const Config & config);
  /** @brief 清空全部轨迹和统计计数 */
  void reset();

  /** @brief 关联候选观测并更新轨迹 @param candidates 当前帧候选 @param timestamp 帧时间戳 @param mode 小符或大符模式 @return 稳定后的轨迹观测 */
  std::vector<BuffObservation> update(
    const std::vector<BuffObservation> & candidates, std::chrono::steady_clock::time_point timestamp,
    BuffMode mode);

  /** @brief 获取主轨迹编号 @return 主轨迹 ID；无主轨迹时为 -1 */
  int primary_track_id() const { return primary_track_id_; }
  /** @brief 获取主轨迹确认切换次数 @return 切换次数 */
  int confirmed_switch_count() const { return confirmed_switch_count_; }
  /** @brief 获取时序关联拒绝次数 @return 拒绝次数 */
  int temporal_reject_count() const { return temporal_reject_count_; }
  /** @brief 获取当前轨迹数 @return 轨迹数 */
  size_t track_count() const { return tracks_.size(); }

private:
  struct Track
  {
    int id = -1;
    int hits = 0;
    bool confirmed = false;
    double angle = 0.0;
    double angular_velocity = 0.0;
    std::chrono::steady_clock::time_point last_update{};
    std::chrono::steady_clock::time_point last_seen{};
    BuffObservation observation;
    std::optional<BuffObservation> pending_recovery;
    int recovery_hits = 0;
  };

  struct Edge
  {
    size_t track_index = 0;
    size_t candidate_index = 0;
    double cost = 0.0;
  };

  /** @brief 获取当前模式允许的最大轨迹数 @param mode 模式 @return 容量 */
  int capacity(BuffMode mode) const;
  /** @brief 外推轨迹在指定时刻的角度 @param track 轨迹 @param timestamp 查询时间 @return 预测角度 */
  double predicted_angle(const Track & track, std::chrono::steady_clock::time_point timestamp) const;
  /** @brief 将候选关键点稳定到轨迹相位 @param candidate 候选观测 @param track 参考轨迹 @return 稳定成功时返回 true */
  bool stabilize_for_track(BuffObservation & candidate, const Track & track) const;
  /** @brief 检查恢复候选是否连续一致 @param candidate 当前候选 @param pending 待确认候选 @return 一致时返回 true */
  bool recovery_consistent(
    const BuffObservation & candidate, const BuffObservation & pending) const;
  /** @brief 使用关联候选更新轨迹 @param track 目标轨迹 @param candidate 候选观测 @param timestamp 时间戳 */
  void update_track(
    Track & track, BuffObservation candidate, std::chrono::steady_clock::time_point timestamp);
  /** @brief 判断候选是否可创建新轨迹 @param candidate 候选观测 @return 可创建时返回 true */
  bool can_spawn(const BuffObservation & candidate) const;
  /** @brief 从候选创建新轨迹 @param candidate 候选观测 @param timestamp 时间戳 */
  void spawn_track(
    BuffObservation candidate, std::chrono::steady_clock::time_point timestamp);
  /** @brief 删除超过保留时限的轨迹 @param timestamp 当前时间 */
  void purge(std::chrono::steady_clock::time_point timestamp);

  Config config_;
  std::vector<Track> tracks_;
  BuffMode mode_ = BuffMode::SMALL;
  bool has_mode_ = false;
  int next_track_id_ = 1;
  int primary_track_id_ = -1;
  int primary_missing_frames_ = 0;
  int confirmed_switch_count_ = 0;
  int temporal_reject_count_ = 0;
};
}  // namespace auto_buff

#endif  // AUTO_BUFF__TRACK_BANK_HPP
