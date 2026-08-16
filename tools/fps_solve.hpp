#ifndef TOOLS__FPS_SOLVE_HPP
#define TOOLS__FPS_SOLVE_HPP

#include <chrono>
#include <cstddef>
#include <deque>

namespace tools
{

class fpsSolve
{
public:
  /** @brief 构造帧率统计器 @param window_size 滑动平均窗口的样本数 */
  explicit fpsSolve(std::size_t window_size = 200);

  /** @brief 加入一个帧时间戳 @param timestamp 当前帧时间 @return 瞬时帧率 */
  double update(
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now());
  /** @brief 获取瞬时帧率 @return 帧率，单位 Hz */
  double get_fps() const;
  /** @brief 获取滑动窗口平均帧率 @return 平均帧率，单位 Hz */
  double get_mean_fps() const;
  /** @brief 清空帧率统计状态 */
  void reset();

private:
  std::size_t window_size_;
  std::deque<double> fps_window_;
  std::chrono::steady_clock::time_point last_timestamp_{};
  double fps_ = 0.0;
  double fps_sum_ = 0.0;
  bool initialized_ = false;
};

}  // namespace tools

#endif  // TOOLS__FPS_SOLVE_HPP
