#include "buff_config.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_buff
{
BuffConfig load_buff_config(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);
  BuffConfig config;
  if (yaml["buff_rune_radius_m"]) config.rune_radius_m = yaml["buff_rune_radius_m"].as<double>();
  if (yaml["buff_small_direction"]) config.small_direction = yaml["buff_small_direction"].as<int>();
  if (yaml["buff_blind_timeout_s"]) config.blind_timeout_s = yaml["buff_blind_timeout_s"].as<double>();
  if (yaml["buff_track_retention_s"]) {
    config.track_retention_s = yaml["buff_track_retention_s"].as<double>();
  }
  if (yaml["buff_fire_full_observation_max_age_s"]) {
    config.fire_full_observation_max_age_s =
      yaml["buff_fire_full_observation_max_age_s"].as<double>();
  }
  if (yaml["buff_direction_confirm_intervals"]) {
    config.direction_confirm_intervals = yaml["buff_direction_confirm_intervals"].as<int>();
  }
  if (yaml["buff_big_speed_phase_window"]) {
    config.big_speed_phase_window = yaml["buff_big_speed_phase_window"].as<int>();
  }
  if (yaml["buff_big_speed_min_span_s"]) {
    config.big_speed_min_span_s = yaml["buff_big_speed_min_span_s"].as<double>();
  }
  if (yaml["buff_big_fit_min_span_s"]) {
    config.big_fit_min_span_s = yaml["buff_big_fit_min_span_s"].as<double>();
  }
  if (yaml["buff_big_fit_min_inlier_ratio"]) {
    config.big_fit_min_inlier_ratio = yaml["buff_big_fit_min_inlier_ratio"].as<double>();
  }
  if (yaml["buff_big_fit_max_rms"]) {
    config.big_fit_max_rms = yaml["buff_big_fit_max_rms"].as<double>();
  }
  if (yaml["buff_big_fit_blend_s"]) {
    config.big_fit_blend_s = yaml["buff_big_fit_blend_s"].as<double>();
  }
  return config;
}
}  // namespace auto_buff
