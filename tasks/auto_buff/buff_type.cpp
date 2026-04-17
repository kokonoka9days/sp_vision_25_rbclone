#include "buff_type.hpp"

#include <algorithm>

#include "tools/logger.hpp"
namespace auto_buff
{
FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
}

FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  if (t != _unlight) exit(-1);
}

PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
: r_center(center), light_num(ts.size())
{
  /// 找出target，并实现【死锁+超时重置】机制
  if (ts.empty()) {
    unsolvable_ = true;
    return;
  }

  auto target_fanblade_it = ts.begin();

  if (last_powerrune.has_value() && !last_powerrune.value().is_unsolve()) {
    auto last_target_center = last_powerrune.value().fanblades[0].center;
    float min_distance = std::numeric_limits<float>::max();
    
    // 寻找距离上一帧目标最近的扇叶
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float distance = cv::norm(it->center - last_target_center);
      if (distance < min_distance) {
        min_distance = distance;
        target_fanblade_it = it;
      }
    }

    // ========== 核心死锁验证逻辑 ==========
    // 计算大符半径（中心 R 标到上一帧靶子中心的距离）
    double radius = cv::norm(last_target_center - r_center);
    
    // 大符相邻两个扇叶的物理距离约为 半径 * 1.17
    // 如果算出来的最短距离依然大于 0.4 倍半径，说明我们死锁的那个靶子在这帧漏检了！
    // 此时视野里即便有别的靶子，也绝对不能切过去，直接判定本帧无法解算。
    if (radius > 10.0 && min_distance > radius * 0.4) {
      unsolvable_ = true;
      return; // 直接 return，触发 detector 的 handle_lose() 累计丢失帧
    }
    // ======================================
  }

  target_fanblade_it->type = _target;
  std::iter_swap(ts.begin(), target_fanblade_it);

  /// 填充FanBlade.angle
  double angle = atan_angle(ts[0].center);
  for (auto & t : ts) {
    t.angle = atan_angle(t.center) - angle;
    if (t.angle < -1e-3) t.angle += CV_2PI;
  }

  /// fanblades调整顺序
  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.angle < b.angle;
  });  // 按照 t.angle 从小到大排序 ts
  const std::vector<double> target_angles = {
    0, 2.0 * CV_PI / 5.0, 4.0 * CV_PI / 5.0, 6.0 * CV_PI / 5.0, 8.0 * CV_PI / 5.0};
  for (int i = 0, j = 0; i < 5 && j < ts.size(); i++) {
    if (std::fabs(ts[j].angle - target_angles[i]) < CV_PI / 5.0)
      fanblades.emplace_back(ts[j++]);
    else
      fanblades.emplace_back(FanBlade(_unlight));
  }
};

double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}
}  // namespace auto_buff