#include <cstdlib>

#include "tasks/auto_buff/buff_target.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main()
{
  auto_buff::BuffConfig first_config;
  first_config.rune_radius_m = 0.61;
  first_config.small_direction = 1;
  first_config.track_retention_s = 0.25;

  auto_buff::BuffConfig second_config;
  second_config.rune_radius_m = 0.83;
  second_config.small_direction = -1;
  second_config.track_retention_s = 0.75;

  const auto_buff::SmallTarget first(first_config);
  const auto_buff::SmallTarget second(second_config);

  CHECK(first.config().rune_radius_m == 0.61);
  CHECK(first.config().small_direction == 1);
  CHECK(first.config().track_retention_s == 0.25);
  CHECK(second.config().rune_radius_m == 0.83);
  CHECK(second.config().small_direction == -1);
  CHECK(second.config().track_retention_s == 0.75);
  return 0;
}
