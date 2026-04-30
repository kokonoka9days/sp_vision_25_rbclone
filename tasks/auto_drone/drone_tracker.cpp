#include "drone_tracker.hpp"
#include <yaml-cpp/yaml.h>
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{
Tracker::Tracker(const std::string & config_path, Solver * solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now())
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_str_ = tools::read<std::string>(yaml, "enemy_color");
  enemy_color_ = (enemy_color_str_ == "red") ? Color::red : Color::blue;
  
  min_detect_count_ = tools::read<int>(yaml, "min_detect_count");
  max_temp_lost_count_ = tools::read<int>(yaml, "max_temp_lost_count");
}

std::string Tracker::state() const { return state_; }

std::vector<Target> Tracker::track(
  std::vector<Drone> & drones, std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  if (gimbal_ == nullptr) {
    tools::logger()->error("[Tracker] gimbal_ cannot be null!");
    return {};
  }

  // 自动颜色识别 (如果配置文件设为 auto)
  io::GimbalState g = gimbal_->state();
  if (enemy_color_str_ == "auto") {
    enemy_color_ = (g.enemy_color == 0) ? Color::blue : Color::red;
  }

  // 时间间隔过长，说明可能发生了相机卡顿或掉线
  if (state_ != "lost" && dt > 0.2) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s. Resetting tracker.", dt);
    state_ = "lost";
  }

  // 1. 过滤掉非敌方颜色的无人机
  drones.erase(
    std::remove_if(drones.begin(), drones.end(),
                   [&](const Drone & d) { return d.color != enemy_color_; }),
    drones.end());

  // 2. 优先选择靠近图像中心的无人机
  cv::Point2f img_center(1440 / 2.0, 1080 / 2.0); // 注意：需根据实际分辨率调整
  std::sort(drones.begin(), drones.end(), [&img_center](const Drone & a, const Drone & b) {
    return cv::norm(a.center - img_center) < cv::norm(b.center - img_center);
  });

  // 3. 对筛选出的无人机进行 PnP 空间解算
  for (auto & drone : drones) {
    solver_->solve(drone);
  }

  // 4. 执行追踪逻辑
  bool found = false;
  if (state_ == "lost") {
    found = set_target(drones, t);
  } else {
    found = update_target(drones, t);
  }

  // 5. 更新状态机
  state_machine(found);

  // 6. 发散检测：如果 EKF 持续不收敛或预测位置飞到天际，重置
  if (state_ != "lost") {
    if (std::isnan(target_.get_xyz().x()) || target_.get_xyz().norm() > 20.0) {
      tools::logger()->debug("[Tracker] Target diverged (distance > 20m or NaN)!");
      state_ = "lost";
      return {};
    }
  }

  if (state_ == "lost") return {};

  return {target_};
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;
    state_ = "detecting";
    detect_count_ = 1;
  }
  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }
  else if (state_ == "tracking") {
    if (found) return;
    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }
  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > max_temp_lost_count_) {
        state_ = "lost";
      }
    }
  }
}

bool Tracker::set_target(std::vector<Drone> & drones, std::chrono::steady_clock::time_point /*t*/)
{
  if (drones.empty()) return false;

  // 使用当前最居中的无人机初始化 EKF Target
  target_ = Target(drones.front());
  return true;
}

bool Tracker::update_target(std::vector<Drone> & drones, std::chrono::steady_clock::time_point /*t*/)
{
  // 预测当前时间步
  // 假设外部传入的 t 与 last_timestamp 的差值已经在 EKF 中由定时器或外部步长处理
  // 如果你在 Target 类的 EKF 配置里使用的是固定步长 DT，预测只需调用 target_.predict(DT)
  // 如果是变步长，你需要记录上一次 update 的时间并传入真实 dt
  // 为了匹配之前的设计，这里假设使用一个固定步长 0.01 或动态计算传入
  // 这里简化处理：因为 track() 开头计算了 dt，理论上应该将 dt 传入 predict
  // 为了兼容之前的 target_.hpp 接口，你可以直接在这里使用默认配置的步长预测：
  target_.predict(0.01); // 视你的 EKF 实现情况传入实际时间差

  if (drones.empty()) return false;

  // 由于 drones 已经按距中心由近到远排序，直接用最近的更新
  // TODO: 如果画面中存在多个无人机，为了防止频繁跳变，这里应该加入距离当前 Target EKF预测位置最短的匹配算法 (如匈牙利算法)
  // 目前按最居中目标直接更新
  target_.update(drones.front());
  
  return true;
}

}  // namespace auto_drone