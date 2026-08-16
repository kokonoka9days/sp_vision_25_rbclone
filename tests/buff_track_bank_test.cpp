#include <chrono>
#include <cmath>
#include <cstdlib>

#include "tasks/auto_buff/buff_track_bank.hpp"
#include "tasks/auto_buff/buff_target.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

namespace
{
auto_buff::BuffObservation observation(double angle)
{
  auto_buff::BuffObservation value;
  value.type = auto_buff::BuffObservationType::FULL;
  value.r_center = {100.0f, 100.0f};
  value.target_center = {
    100.0f + static_cast<float>(50.0 * std::cos(angle)),
    100.0f + static_cast<float>(50.0 * std::sin(angle))};
  value.target_points = {{95, 45}, {105, 45}, {105, 55}, {95, 55}};
  value.raw_target_points = value.target_points;
  value.angle = angle;
  value.confidence = 0.95f;
  value.min_keypoint_confidence = 0.95f;
  return value;
}
}  // namespace

int main()
{
  auto_buff::BuffTrackBank::Config config;
  config.confirm_hits = 2;
  auto_buff::BuffTrackBank bank(config);
  const auto start = std::chrono::steady_clock::now();

  CHECK(bank.update({observation(0.0)}, start, auto_buff::BuffMode::SMALL).empty());
  const auto confirmed =
    bank.update({observation(0.01)}, start + std::chrono::milliseconds(10), auto_buff::BuffMode::SMALL);
  CHECK(confirmed.size() == 1);
  CHECK(confirmed.front().primary);
  CHECK(confirmed.front().track_status == auto_buff::BuffTrackStatus::CONFIRMED);
  CHECK(bank.track_count() == 1);

  auto_buff::PhaseDirectionTracker direction(3);
  direction.update(0.00);
  direction.update(0.01);
  direction.update(0.02);
  direction.update(0.03);
  CHECK(direction.ready());
  CHECK(direction.direction() == 1);
  direction.update(0.02);
  direction.update(0.01);
  CHECK(direction.direction() == 1);
  direction.reset();
  CHECK(!direction.ready());

  CHECK(bank.update({}, start + std::chrono::milliseconds(20), auto_buff::BuffMode::SMALL).empty());
  CHECK(bank.track_count() == 1);

  // A mode transition resets the bank and requires confirmation again.
  CHECK(
    bank.update({observation(0.02)}, start + std::chrono::milliseconds(30), auto_buff::BuffMode::BIG)
      .empty());
  CHECK(bank.track_count() == 1);
  return 0;
}
