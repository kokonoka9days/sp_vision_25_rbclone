#include "buff_type.hpp"

#include <algorithm>
#include <limits>

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
  /// 找出target，并实现【死锁验证+击打跳变】融合机制
  if (ts.empty()) {
    unsolvable_ = true;
    return;
  }

  auto target_fanblade_it = ts.begin();

  if (last_powerrune.has_value() && !last_powerrune.value().is_unsolve()) {
    auto last_target_center = last_powerrune.value().fanblades[0].center;
    float min_distance = std::numeric_limits<float>::max();
    
    // 寻找画面中距离上一帧目标最近的扇叶
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float distance = cv::norm(it->center - last_target_center);
      if (distance < min_distance) {
        min_distance = distance;
        target_fanblade_it = it;
      }
    }

    // ========== 融合：追踪与击打跳变验证逻辑 ==========
    // 计算大符半径（中心 R 标到上一帧靶子中心的距离）
    double radius = cv::norm(last_target_center - r_center);
    
    // 确保大符半径合理，排除刚启动时的畸形值（此处 10.0 为像素距离阈值，可依实际相机焦距微调）
    if (radius > 10.0) { 
      if (min_distance <= radius * 0.4) {
        // 1. 正常连续追踪：当前找到的扇叶就在上一帧位置附近。
        // target_fanblade_it 仍正确指向当前帧的同一目标，无需额外处理。
      } 
      else if (min_distance > radius * 0.8 && min_distance < radius * 1.5) {
        // 2. 合法击打跳变：发生切换时，新扇叶与老扇叶的距离是弦长。
        // 相邻扇叶的物理弦长 L = 2 * R * sin(72°/2) ≈ 1.175 * R。
        // 设定 0.8R ~ 1.5R 的宽容区间，容忍 YOLO 的检测框抖动。
        // 此时 target_fanblade_it 自然指向了新亮起的扇叶，放行。
        tools::logger()->debug("[PowerRune] Target Switched! Jump distance: {:.2f}R", min_distance / radius);
      } 
      else {
        // 3. 死锁或误检：距离不合理。
        // 例如跳跃距离跨越了对角线(>1.5R)，或者处在尴尬距离(0.4R~0.8R)说明框到了背景杂光。
        // 说明我们想要的目标（或者下一个该打的目标）漏检了。
        unsolvable_ = true;
        return; // 直接丢弃本帧解算，交给后端的 Predictor / EKF 进行盲推
      }
    }
    // ==================================================
  }

  // 确立 Target 属性并交换到首位
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