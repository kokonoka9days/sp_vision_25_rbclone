#ifndef AUTO_BUFF__CONFIG_HPP
#define AUTO_BUFF__CONFIG_HPP

#include <string>

namespace auto_buff
{
struct BuffConfig
{
  double rune_radius_m = 0.700;
  int small_direction = 0;
  double blind_timeout_s = 0.100;
  double track_retention_s = 0.400;
  double fire_full_observation_max_age_s = 0.030;
  int direction_confirm_intervals = 3;
  int big_speed_phase_window = 7;
  double big_speed_min_span_s = 0.030;
  double big_fit_min_span_s = 1.0;
  double big_fit_min_inlier_ratio = 0.75;
  double big_fit_max_rms = 0.18;
  double big_fit_blend_s = 0.30;
};

/** @brief 从 YAML 文件加载能量机关配置 @param config_path 配置文件路径 @return 完整配置；缺失字段使用默认值 */
BuffConfig load_buff_config(const std::string & config_path);
}  // namespace auto_buff

#endif  // AUTO_BUFF__CONFIG_HPP
