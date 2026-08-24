#include <cstdlib>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "tasks/auto_buff/rune_system.hpp"

#define CHECK(condition) do { if (!(condition)) std::abort(); } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 2);
  auto_buff::RuneSystem rune(argv[1]);
  const auto now = std::chrono::steady_clock::now();

  auto command = rune.process(
    {}, now, Eigen::Quaterniond::Identity(), auto_buff::BuffMode::SMALL,
    auto_buff::EnemyColor::RED, 24.5f);
  CHECK(!command.found && !command.fire);
  CHECK(rune.debug_snapshot().failure_reason == auto_buff::RuneFailureReason::EMPTY_IMAGE);

  const cv::Mat blank = cv::Mat::zeros(480, 640, CV_8UC3);
  command = rune.process(
    blank, now, Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0), auto_buff::BuffMode::SMALL,
    auto_buff::EnemyColor::RED, 24.5f);
  CHECK(!command.found && !command.fire);
  CHECK(rune.debug_snapshot().failure_reason == auto_buff::RuneFailureReason::INVALID_POSE);

  command = rune.process(
    blank, now, Eigen::Quaterniond::Identity(), auto_buff::BuffMode::SMALL,
    static_cast<auto_buff::EnemyColor>(9), 24.5f);
  CHECK(!command.found && !command.fire);
  CHECK(rune.debug_snapshot().failure_reason == auto_buff::RuneFailureReason::INVALID_COLOR);

  command = rune.process(
    blank, now, Eigen::Quaterniond::Identity(), auto_buff::BuffMode::SMALL,
    auto_buff::EnemyColor::RED, 24.5f);
  CHECK(!command.found && !command.fire);
  CHECK(rune.debug_snapshot().failure_reason == auto_buff::RuneFailureReason::DETECTION_EMPTY);

  command = rune.process(
    blank, now, Eigen::Quaterniond::Identity(), auto_buff::BuffMode::BIG,
    auto_buff::EnemyColor::RED, 24.5f);
  CHECK(!command.found && !command.fire);
  CHECK(rune.debug_snapshot().failure_reason == auto_buff::RuneFailureReason::MODE_CHANGED);
  return 0;
}
