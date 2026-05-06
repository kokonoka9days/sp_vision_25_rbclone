#include "buff_type.hpp"
#include <algorithm>
#include <limits>
#include "tools/logger.hpp"

namespace auto_buff
{

// ======================= 扇叶 (FanBlade) =======================

FanBlade::FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
}

FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  // 仅允许用来构造未点亮的虚拟占位扇叶
  if (t != _unlight) exit(-1); 
}

// ====================== 大神符 (PowerRune) ======================

PowerRune::PowerRune(std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
: r_center(center), light_num(ts.size())
{
  if (ts.empty()) {
    unsolvable_ = true;
    return;
  }

  auto target_it = ts.begin(); // 默认首个扇叶为追踪目标

  // --- 步骤 1: 目标追踪与抗跳变逻辑 ---
  if (last_powerrune.has_value() && !last_powerrune.value().is_unsolve()) {
    auto last_target = last_powerrune.value().fanblades[0].center;
    auto last_r = last_powerrune.value().r_center; 

    // 补偿相机运动：将上一帧靶标对齐到当前画面坐标系
    cv::Point2f aligned_last_target = last_target + (r_center - last_r);
    
    // 寻找距离上一帧靶心最近的当前扇叶
    float min_distance = std::numeric_limits<float>::max();
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float dist = cv::norm(it->center - aligned_last_target);
      if (dist < min_distance) {
        min_distance = dist;
        target_it = it;
      }
    }

    // 计算当前帧的真实半径
    double radius = cv::norm(target_it->center - r_center);
    static int fake_jump_count = 0; 

    // 排除程序刚启动时的畸形半径
    if (radius > 10.0) { 
      if (min_distance <= radius * 0.4) {
        // [正常追踪] 距离波动极小，追踪稳定，清空跳变冷却计数
        fake_jump_count = 0;
      } 
      else if ((min_distance > radius * 0.9 && min_distance < radius * 1.45) || 
               (min_distance > radius * 1.5 && min_distance < radius * 2.2)) {
        // [跳变嫌疑] 处于合法的击打跳变距离，增加冷却计数
        fake_jump_count++;
        
        if (fake_jump_count >= 3) {
          tools::logger()->debug("[PowerRune] 目标合法跳变! 弦长比: {:.2f}R", min_distance / radius);
        } else {
          // 冷却期(1~2帧)内丢弃当前帧，触发 Target 基类的 EKF 盲推机制以防闪烁
          unsolvable_ = true;
          return;
        }
      } 
      else {
        // [异常跳变] 不在合法跳变区间的突变，直接丢弃
        unsolvable_ = true;
        return; 
      }
    }
  }

  // --- 步骤 2: 确立最终目标并移至容器首位 ---
  target_it->type = _target;
  std::iter_swap(ts.begin(), target_it);

  // --- 步骤 3: 以目标扇叶为基准 0 度，计算其余扇叶相对极角并排序 ---
  double base_angle = atan_angle(ts[0].center);
  for (auto & t : ts) {
    t.angle = atan_angle(t.center) - base_angle;
    if (t.angle < -1e-3) t.angle += CV_2PI;
  }
  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) { 
    return a.angle < b.angle; 
  });
  
  // --- 步骤 4: 补齐 5 个扇叶位 (无真实扇叶的位置用 _unlight 占位) ---
  const std::vector<double> standard_angles = { 0, 2.0 * CV_PI / 5.0, 4.0 * CV_PI / 5.0, 6.0 * CV_PI / 5.0, 8.0 * CV_PI / 5.0 };
  for (int i = 0, j = 0; i < 5 && j < ts.size(); i++) {
    if (std::fabs(ts[j].angle - standard_angles[i]) < CV_PI / 5.0)
      fanblades.emplace_back(ts[j++]);
    else
      fanblades.emplace_back(FanBlade(_unlight));
  }
}

// 辅助函数：计算极角并将其严格映射到 [0, 2π] 区间
double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI; 
}

}  // namespace auto_buff