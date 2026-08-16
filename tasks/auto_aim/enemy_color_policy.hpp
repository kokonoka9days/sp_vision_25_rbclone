#ifndef AUTO_AIM__ENEMY_COLOR_POLICY_HPP
#define AUTO_AIM__ENEMY_COLOR_POLICY_HPP

#include <cstdint>
#include <list>

#include "armor.hpp"

namespace auto_aim
{
enum class EnemyColorPolicy
{
  Standard,
  Sentry
};

/** @brief 将电控颜色字段转换为视觉敌方颜色 @param policy 颜色编码策略 @param enemy_color 电控颜色值 @return 视觉颜色枚举 */
inline Color enemy_color_from_gimbal(EnemyColorPolicy policy, std::uint8_t enemy_color)
{
  if (policy == EnemyColorPolicy::Sentry) {
    return enemy_color == 0 ? Color::blue : Color::red;
  }
  return enemy_color == 0 ? Color::red : Color::blue;
}

/** @brief 按敌方颜色原地过滤装甲板列表 @param armors 待过滤列表 @param enemy_color 敌方颜色 @param use_enemy_color 是否启用颜色过滤 */
inline void filter_enemy_armors(
  std::list<Armor> & armors, Color enemy_color, bool use_enemy_color)
{
  if (!use_enemy_color) return;
  armors.remove_if([enemy_color](const Armor & armor) { return armor.color != enemy_color; });
}
}  // namespace auto_aim

#endif  // AUTO_AIM__ENEMY_COLOR_POLICY_HPP
