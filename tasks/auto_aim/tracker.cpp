#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver * solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_str_ = yaml["enemy_color"].as<std::string>();
  enemy_color_ = (enemy_color_str_ == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;

  last_cam_is_short = true;
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::sb_track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t,bool cam_is_short, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  if(gimbal_ == nullptr) {
    tools::logger()->error("[Tracker] gimbal_不能为空指针，请先调用set_gimbal()设置云台指针");
    return {};
  }
  io::GimbalState g = gimbal_->state();
  if(enemy_color_str_ == "auto") enemy_color_ = (g.enemy_color == 0) ? Color::blue : Color::red;

  target_.cam_is_short = cam_is_short;

  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }
  else {
    found = update_target(armors, t);
  }

  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  if (
  std::accumulate(
    target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
  (0.4 * target_.ekf().window_size)) {
      tools::logger()->debug("[Target] Bad Converge Found!");
      state_ = "lost";
      return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool cam_is_short, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;
  if(gimbal_ == nullptr) {
    tools::logger()->error("[Tracker] gimbal_不能为空指针，请先调用set_gimbal()设置云台指针");
    return {};
  }
  io::GimbalState g = gimbal_->state();
  if(enemy_color_str_ == "auto") enemy_color_ = (g.enemy_color == 0) ?   Color::red :Color::blue;

  target_.cam_is_short = cam_is_short;

  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  bool found = 0;
  if(g.mode == 1)
  {
    if (state_ == "lost") {
        found = set_target(armors, t);
    }
    else {
      found = update_target(armors, t);
    }
  }else if(g.mode == 0){
    if (state_ == "lost") {
        found = set_target(armors, t);
    }
    else {
      if(target_.name == armors.front().name 
        && target_.armor_type == armors.front().type)
      {
        found = update_target(armors, t);
      }else{
        found = set_target(armors, t);
        state_ = "detecting";
        detect_count_ = 1;
      }
    }
  }

  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    state_ = "lost";
    return {};
  }

  if (
  std::accumulate(
    target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
  (0.4 * target_.ekf().window_size)) {
      tools::logger()->debug("[Target] Bad Converge Found!");
      state_ = "lost";
      return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }
  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }
  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }
  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  
  }

  if (state_ == "lost") return {switch_target, {}};  

  std::list<Target> targets = {target_};
  return {switch_target, targets};
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
  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }
  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_->solve(armor);

  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }
  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }
  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }
  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

// ================= [核心修改处] ====================
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);
  
  bool found = false;
  std::vector<Armor> matched_armors;

  // 收集当前视野里属于目标车辆的【所有】有效装甲板
  for (auto & armor : armors) {
    if (armor.name == target_.name && armor.type == target_.armor_type) {
      solver_->solve(armor);
      matched_armors.push_back(armor);
      found = true;
    }
  }

  // 将多块装甲板同时喂给 Target，内部使用匈牙利全局匹配
  if (found) {
    target_.update(matched_armors); 
  }

  return found;
}

}  // namespace auto_aim
