#include <cstdlib>

#include "tasks/auto_aim/tracking/enemy_color_policy.hpp"

#define CHECK(condition)          \
  do {                            \
    if (!(condition)) std::abort(); \
  } while (false)

int main()
{
  using auto_aim::Color;
  using auto_aim::EnemyColorPolicy;
  using auto_aim::enemy_color_from_gimbal;

  CHECK(enemy_color_from_gimbal(EnemyColorPolicy::Standard, 0) == Color::red);
  CHECK(enemy_color_from_gimbal(EnemyColorPolicy::Standard, 1) == Color::blue);
  CHECK(enemy_color_from_gimbal(EnemyColorPolicy::Sentry, 0) == Color::blue);
  CHECK(enemy_color_from_gimbal(EnemyColorPolicy::Sentry, 1) == Color::red);

  auto_aim::Armor red_armor;
  auto_aim::Armor blue_armor;
  red_armor.color = Color::red;
  blue_armor.color = Color::blue;

  std::list<auto_aim::Armor> filtered{red_armor, blue_armor};
  auto_aim::filter_enemy_armors(filtered, Color::red, true);
  CHECK(filtered.size() == 1);
  CHECK(filtered.front().color == Color::red);

  std::list<auto_aim::Armor> unfiltered{red_armor, blue_armor};
  auto_aim::filter_enemy_armors(unfiltered, Color::red, false);
  CHECK(unfiltered.size() == 2);

  return 0;
}
