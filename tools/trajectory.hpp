#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct TrajectoryV1
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  // 不考虑空气阻力
  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m

  TrajectoryV1() = default;
  TrajectoryV1(const double v0, const double d, const double h);
};

struct TrajectoryV2 : TrajectoryV1{
  TrajectoryV2(const double v0, const double d, const double h);
};

// 新增 V3 声明
struct TrajectoryV3 : TrajectoryV1{
  TrajectoryV3(const double v0, const double d, const double h);
};

// 将默认使用的是别名切换为 V3（如果测试通过的话）
using Trajectory = TrajectoryV3;

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP