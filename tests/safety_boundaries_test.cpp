#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "tasks/auto_aim/aiming/planner/planner.hpp"
#include "tasks/auto_aim/geometry/solver.hpp"
#include "tasks/omniperception/decider.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 2);

  omniperception::Decider decider(argv[1]);
  std::vector<omniperception::DetectionResult> detections{
    omniperception::DetectionResult{{}, std::chrono::steady_clock::now(), 0.0, 0.0}};
  decider.sort(detections);
  CHECK(detections.empty());

  const std::vector<cv::Point2f> points{{0, 0}, {10, 0}, {10, 5}, {0, 5}};
  std::list<auto_aim::Armor> prioritized{
    auto_aim::Armor(6, 1.0F, cv::Rect(0, 0, 10, 5), points)};
  decider.set_mode(1);
  decider.set_priority(prioritized);
  CHECK(prioritized.front().priority == auto_aim::ArmorPriority::forth);
  decider.set_mode(2);
  decider.set_priority(prioritized);
  CHECK(prioritized.front().priority == auto_aim::ArmorPriority::first);
  std::list<auto_aim::Armor> invalid_mode_armor{auto_aim::Armor{}};
  decider.set_mode(99);
  decider.set_priority(invalid_mode_armor);
  CHECK(invalid_mode_armor.front().priority == auto_aim::ArmorPriority::fifth);

  auto_aim::Solver solver(argv[1]);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto_aim::Armor armor(
    0, 1.0F, cv::Rect{},
    std::vector<cv::Point2f>{{nan, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}});
  armor.xyz_in_world = Eigen::Vector3d(1.0, 2.0, 3.0);
  CHECK(!solver.try_solve(armor));
  CHECK(armor.xyz_in_world == Eigen::Vector3d(1.0, 2.0, 3.0));

  CHECK(auto_aim::valid_shoot_offset(-50));
  CHECK(auto_aim::valid_shoot_offset(49));
  CHECK(!auto_aim::valid_shoot_offset(-51));
  CHECK(!auto_aim::valid_shoot_offset(50));
  return 0;
}
