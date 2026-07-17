#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "tasks/auto_buff/buff_track_bank.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

cv::Point2f rotate(const cv::Point2f & point, const cv::Point2f & center, double angle)
{
  const float c = static_cast<float>(std::cos(angle));
  const float s = static_cast<float>(std::sin(angle));
  const auto relative = point - center;
  return center + cv::Point2f(c * relative.x - s * relative.y, s * relative.x + c * relative.y);
}

auto_buff::BuffObservation make_observation(
  double angle, auto_buff::BuffObservationType type = auto_buff::BuffObservationType::FULL,
  float confidence = 0.95f)
{
  const cv::Point2f rune_center{500.0f, 500.0f};
  const std::vector<cv::Point2f> base_target{
    {700.0f, 490.0f}, {710.0f, 500.0f}, {700.0f, 510.0f}, {690.0f, 500.0f}};
  const std::vector<cv::Point2f> base_fan{
    {590.0f, 490.0f}, {610.0f, 490.0f}, {610.0f, 510.0f}, {590.0f, 510.0f}};

  auto_buff::BuffObservation observation;
  observation.type = type;
  observation.r_center = rune_center;
  observation.center_source = auto_buff::RuneCenterSource::DETECTED;
  observation.angle = angle;
  observation.confidence = confidence;
  observation.min_keypoint_confidence = confidence;
  observation.quad_quality = 1.0;
  observation.pair_distance_ratio = 0.5;

  if (type != auto_buff::BuffObservationType::FAN_ONLY) {
    for (const auto & point : base_target) {
      observation.target_points.push_back(rotate(point, rune_center, angle));
    }
    observation.raw_target_points = observation.target_points;
    observation.target_center = rotate({700.0f, 500.0f}, rune_center, angle);
    observation.target_center_observed = true;
  }
  if (type != auto_buff::BuffObservationType::TARGET_ONLY) {
    for (const auto & point : base_fan) {
      observation.fan_points.push_back(rotate(point, rune_center, angle));
    }
    observation.raw_fan_points = observation.fan_points;
    observation.fan_center = rotate({600.0f, 500.0f}, rune_center, angle);
    observation.fan_center_observed = true;
  }
  return observation;
}

bool require(bool condition, const char * message)
{
  if (condition) return true;
  std::cerr << "FAILED: " << message << '\n';
  return false;
}
}  // namespace

int main()
{
  auto_buff::BuffTrackBank::Config config;
  config.confirm_hits = 2;
  config.recovery_hits = 2;
  config.control_blind_timeout_s = 0.100;
  config.retention_timeout_s = 0.400;
  auto_buff::BuffTrackBank bank(config);
  const auto start = Clock::now();
  bool ok = true;

  auto output = bank.update({make_observation(0.0)}, start, auto_buff::BuffMode::SMALL);
  ok &= require(output.empty(), "small track must be tentative on first frame");
  output = bank.update(
    {make_observation(0.02)}, start + std::chrono::milliseconds(20),
    auto_buff::BuffMode::SMALL);
  ok &= require(output.size() == 1, "small track confirms on second frame");
  const int small_id = output.empty() ? -1 : output.front().track_id;

  output = bank.update(
    {make_observation(0.04), make_observation(1.6)}, start + std::chrono::milliseconds(40),
    auto_buff::BuffMode::SMALL);
  ok &= require(output.size() == 1, "small capacity remains one with an extra false candidate");
  ok &= require(output.front().track_id == small_id, "single false candidate must not switch id");

  const auto big_start = start + std::chrono::seconds(1);
  output = bank.update(
    {make_observation(0.0), make_observation(auto_buff::RUNE_SLOT_ANGLE)}, big_start,
    auto_buff::BuffMode::BIG);
  ok &= require(output.empty(), "big tracks start tentative");
  output = bank.update(
    {make_observation(0.02), make_observation(auto_buff::RUNE_SLOT_ANGLE + 0.02)},
    big_start + std::chrono::milliseconds(20), auto_buff::BuffMode::BIG);
  ok &= require(output.size() == 2, "big confirms two independent tracks");
  const int old_primary = output.empty() ? -1 : output.front().track_id;
  const int backup_id = output.size() < 2 ? -1 : output[1].track_id;
  const double backup_angle = output.size() < 2 ? auto_buff::RUNE_SLOT_ANGLE : output[1].angle;

  output = bank.update(
    {make_observation(backup_angle + 0.02)}, big_start + std::chrono::milliseconds(40),
    auto_buff::BuffMode::BIG);
  ok &= require(output.size() == 1, "big accepts a long-running single group");
  ok &= require(!output.front().primary, "one missing primary frame is held against flicker");
  output = bank.update(
    {make_observation(backup_angle + 0.04)}, big_start + std::chrono::milliseconds(60),
    auto_buff::BuffMode::BIG);
  ok &= require(output.front().track_id == backup_id, "visible backup takes over within one frame");
  ok &= require(output.front().primary, "backup is promoted to primary");
  ok &= require(output.front().track_id != old_primary, "hit primary is not kept over visible backup");

  bank.update({}, big_start + std::chrono::milliseconds(120), auto_buff::BuffMode::BIG);
  ok &= require(bank.track_count() == 2, "tracks survive a short dropout");
  bank.update({}, big_start + std::chrono::milliseconds(480), auto_buff::BuffMode::BIG);
  ok &= require(bank.track_count() == 0, "tracks expire after retention timeout");

  const auto recovery_start = start + std::chrono::seconds(2);
  bank.update({make_observation(0.0)}, recovery_start, auto_buff::BuffMode::SMALL);
  output = bank.update(
    {make_observation(0.02)}, recovery_start + std::chrono::milliseconds(20),
    auto_buff::BuffMode::SMALL);
  const int recovery_id = output.empty() ? -1 : output.front().track_id;
  output = bank.update(
    {make_observation(1.0)}, recovery_start + std::chrono::milliseconds(40),
    auto_buff::BuffMode::SMALL);
  ok &= require(output.empty(), "single large jump is isolated");
  output = bank.update(
    {make_observation(0.04)}, recovery_start + std::chrono::milliseconds(60),
    auto_buff::BuffMode::SMALL);
  ok &= require(output.size() == 1 && output.front().track_id == recovery_id,
    "normal observation resumes without state pollution");
  bank.update(
    {make_observation(1.0)}, recovery_start + std::chrono::milliseconds(80),
    auto_buff::BuffMode::SMALL);
  output = bank.update(
    {make_observation(1.02)}, recovery_start + std::chrono::milliseconds(100),
    auto_buff::BuffMode::SMALL);
  ok &= require(output.size() == 1 && output.front().track_id == recovery_id,
    "two coherent shifted frames re-anchor the same track");

  bank.reset();
  bank.update(
    {make_observation(0.0, auto_buff::BuffObservationType::TARGET_ONLY)},
    start + std::chrono::seconds(3), auto_buff::BuffMode::BIG);
  ok &= require(bank.track_count() == 0, "partial observation cannot spawn a track");

  if (!ok) return 1;
  std::cout << "auto_buff_track_bank_test passed\n";
  return 0;
}
