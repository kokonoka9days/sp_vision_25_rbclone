#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct TrajectoryV1
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  /** @brief 默认构造未初始化的轨迹结果 */
  TrajectoryV1() = default;
  /** @brief 计算不考虑空气阻力的弹道 @param v0 弹丸初速度，单位 m/s @param d 目标水平距离，单位 m @param h 目标竖直高度，单位 m */
  TrajectoryV1(const double v0, const double d, const double h);
};

struct TrajectoryV2 : TrajectoryV1{
  /** @brief 计算包含空气阻力修正的弹道 @param v0 弹丸初速度，单位 m/s @param d 目标水平距离，单位 m @param h 目标竖直高度，单位 m */
  TrajectoryV2(const double v0, const double d, const double h);
};


using Trajectory = TrajectoryV2;

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
