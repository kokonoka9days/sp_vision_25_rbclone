#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

namespace tools
{
class Exiter
{
public:
  /** @brief 注册进程退出信号处理器 */
  Exiter();

  /** @brief 查询是否收到退出信号 @return 需要退出时返回 true */
  bool exit() const;
};

}  // namespace tools

#endif  // TOOLS__EXITER_HPP
