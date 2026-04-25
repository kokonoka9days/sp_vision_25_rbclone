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
    auto last_r_center = last_powerrune.value().r_center; // 获取上一帧的R标中心

    // ================== 核心修复 1：相机运动补偿 ==================
    // R 标是物理静止的（相对于大符支架），它在画面中的像素移动，纯粹是因为云台转了。
    // 计算 R 标的位移，就是相机的像素位移。
    cv::Point2f camera_shift = r_center - last_r_center;
    
    // 把上一帧的靶子中心，平移对齐到当前帧的画面坐标系下
    cv::Point2f aligned_last_target = last_target_center + camera_shift;
    // ==============================================================

    float min_distance = std::numeric_limits<float>::max();
    
    // 寻找画面中距离【对齐后的上一帧目标】最近的扇叶
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float distance = cv::norm(it->center - aligned_last_target); // 用对齐后的坐标计算
      if (distance < min_distance) {
        min_distance = distance;
        target_fanblade_it = it;
      }
    }

    // ================== 核心修复 2：计算真实的当前帧半径 ==================
    // 绝对不能跨帧计算！直接使用当前帧选中的扇叶中心到当前帧 R 标的距离
    double radius = cv::norm(target_fanblade_it->center - r_center);
    // tools::logger()->debug("[PowerRune] radius: {:.2f}", radius);
    // ======================================================================
    
    // 确保大符半径合理，排除刚启动时的畸形值
    if (radius > 10.0) { 
      if (min_distance <= radius * 0.4) {
        // 1. 正常连续追踪
      } 
      // 严格限定跳变的弦长比例
      else if ((min_distance > radius * 0.9 && min_distance < radius * 1.45) || 
               (min_distance > radius * 1.5 && min_distance < radius * 2.2)) {
        // 2. 合法击打跳变
        tools::logger()->debug("[PowerRune] Valid Target Switched! Jump distance: {:.2f}R", min_distance / radius);
      } 
      else {
        // 3. 错帧、反光干扰引起的“伪跳变” 
        tools::logger()->debug("[PowerRune] FAKE Jump Blocked! min_d: {:.2f}, radius: {:.2f}, ratio: {:.2f}R", min_distance, radius, min_distance / radius);
        unsolvable_ = true;
        return; 
      }
    }
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