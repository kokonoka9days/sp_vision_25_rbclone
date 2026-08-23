#ifndef AUTO_AIM__VOTER_HPP
#define AUTO_AIM__VOTER_HPP

#include <vector>

#include "../model/armor.hpp"

namespace auto_aim
{

class Voter
{
public:
  /** @brief 构造并清零装甲板属性投票器 */
  Voter();
  /** @brief 为一组装甲板属性投一票 @param color 颜色 @param name 名称 @param type 尺寸类型 */
  void vote(const Color color, const ArmorName name, const ArmorType type);
  /** @brief 查询一组属性的票数 @param color 颜色 @param name 名称 @param type 尺寸类型 @return 累计票数 */
  std::size_t count(const Color color, const ArmorName name, const ArmorType type);

private:
  std::vector<std::size_t> count_;
  /** @brief 将属性组合映射到计数数组下标 @param color 颜色 @param name 名称 @param type 尺寸类型 @return 数组下标 */
  std::size_t index(const Color color, const ArmorName name, const ArmorType type) const;
};
}  // namespace auto_aim

#endif
