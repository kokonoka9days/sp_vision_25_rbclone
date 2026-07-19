#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
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

  const auto estimator = yaml["estimator"];
  const auto read_double = [&](const char * key, double fallback) {
    return estimator && estimator[key] ? estimator[key].as<double>() : fallback;
  };
  const auto read_int = [&](const char * key, int fallback) {
    return estimator && estimator[key] ? estimator[key].as<int>() : fallback;
  };
  const auto read_vector = [&](const char * key, const Eigen::Vector3d & fallback) {
    if (!estimator || !estimator[key]) return fallback;
    const auto values = estimator[key].as<std::vector<double>>();
    if (values.size() != 3) return fallback;
    return Eigen::Vector3d(values[0], values[1], values[2]);
  };
  estimator_config_.iterations = read_int("iterations", estimator_config_.iterations);
  estimator_config_.common_acceleration =
    read_vector("common_acceleration", estimator_config_.common_acceleration);
  estimator_config_.outpost_acceleration =
    read_vector("outpost_acceleration", estimator_config_.outpost_acceleration);
  estimator_config_.common_yaw_acceleration =
    read_double("common_yaw_acceleration", estimator_config_.common_yaw_acceleration);
  estimator_config_.outpost_yaw_acceleration =
    read_double("outpost_yaw_acceleration", estimator_config_.outpost_yaw_acceleration);
  estimator_config_.radius_random_walk =
    read_double("radius_random_walk", estimator_config_.radius_random_walk);
  estimator_config_.height_random_walk =
    read_double("height_random_walk", estimator_config_.height_random_walk);
  estimator_config_.roll_pitch_random_walk =
    read_double("roll_pitch_random_walk", estimator_config_.roll_pitch_random_walk);
  estimator_config_.outpost_height_random_walk =
    read_double("outpost_height_random_walk", estimator_config_.outpost_height_random_walk);
  estimator_config_.armor_match_gate =
    read_double("armor_match_gate", estimator_config_.armor_match_gate);
  estimator_config_.armor_match_gate_not_all_init = read_double(
    "armor_match_gate_not_all_init", estimator_config_.armor_match_gate_not_all_init);
  estimator_config_.armor_match_center_weight =
    read_double("armor_match_center_weight", estimator_config_.armor_match_center_weight);
  estimator_config_.armor_match_angle_weight =
    read_double("armor_match_angle_weight", estimator_config_.armor_match_angle_weight);
  estimator_config_.armor_match_perimeter_weight =
    read_double("armor_match_perimeter_weight", estimator_config_.armor_match_perimeter_weight);
  estimator_config_.uvl_pixel_sigma_ratio =
    read_double("uvl_pixel_sigma_ratio", estimator_config_.uvl_pixel_sigma_ratio);
  estimator_config_.uvl_length_sigma_ratio =
    read_double("uvl_length_sigma_ratio", estimator_config_.uvl_length_sigma_ratio);
  estimator_config_.uvl_angle_sigma =
    read_double("uvl_angle_sigma", estimator_config_.uvl_angle_sigma);
  estimator_config_.depth_difference_sigma =
    read_double("depth_difference_sigma", estimator_config_.depth_difference_sigma);

  last_cam_is_short = true;
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::sb_track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t,bool cam_is_short, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // TODO
  if(gimbal_ == nullptr) {
    tools::logger()->error("[Tracker] gimbal_不能为空指针，请先调用set_gimbal()设置云台指针");
    return {};
  }
  io::GimbalState g = gimbal_->state();
  if(enemy_color_str_ == "auto") enemy_color_ = (g.enemy_color == 0) ? Color::blue : Color::red;

  target_.cam_is_short = cam_is_short;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
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

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
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

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  // armors.sort(
  //   [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found = 0;

  static uint8_t last_mode = g.mode;
  bool mode_switch_0to1 = (last_mode == 0 && g.mode == 1);
  //按下右键时，mouse为1则跟随上一次的目标，不按则瞄准最近的装甲板
  if(!mode_switch_0to1)
  {
    if (state_ == "lost") {
        found = set_target(armors, t);
        // tools::logger()->debug("按下右键，只选择正在跟踪的装甲板，跳过其他兵种，直至丢跟踪，初始化跟踪类型为 {}", ARMOR_NAMES[armors.front().name]);
    }
    else {
      found = update_target(armors, t);
    }
  }else {
    if (state_ == "lost") {
        found = set_target(armors, t);
    }
    else if (armors.empty()) {
      found = update_target(armors, t);
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
        // tools::logger()->debug("不按右键，默认选中离图像中心最近的兵种，切换至： {}, 跟踪器重置, 置信度:{:.3f}", ARMOR_NAMES[armors.front().name],armors.front().confidence );
      }

     
    }
  }
  last_mode = g.mode;
  // found = set_target(armors, t);

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    // tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  

  std::list<Target> targets = {target_};
  return targets;
}


std::list<Target> Tracker::test_track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool cam_is_short, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;
  
 
  target_.cam_is_short = cam_is_short;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  // armors.sort(
  //   [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found = 0;
  
  if (state_ == "lost") {
    found = set_target(armors, t);
    // tools::logger()->debug("按下右键，只选择正在跟踪的装甲板，跳过其他兵种，直至丢跟踪，初始化跟踪类型为 {}", ARMOR_NAMES[armors.front().name]);
  }
  else {
    found = update_target(armors, t);
  }
  
  // found = set_target(armors, t);

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    // tools::logger()->debug("[Tracker] Target diverged!");
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

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
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
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

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
        //前哨站的temp_lost_count需要设置的大一些
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
  const auto context = solver_->camera_context();
  const cv::Point2f principal_point(
    context.camera_matrix(0, 2), context.camera_matrix(1, 2));
  armors.sort([&](const Armor & left, const Armor & right) {
    if (left.priority != right.priority) return left.priority < right.priority;
    return cv::norm(left.center - principal_point) < cv::norm(right.center - principal_point);
  });
  for (auto & armor : armors) {
    if (!solver_->solve(armor)) continue;
    target_ = Target(armor, t, estimator_config_);
    return true;
  }
  return false;
}
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);
  const auto context = solver_->camera_context();

  struct VisibleArmor
  {
    int id;
    double facing;
    std::array<cv::Point2f, 4> points;
  };
  std::vector<VisibleArmor> visible;
  const auto poses = target_.armor_pose_list();
  for (int id = 0; id < static_cast<int>(poses.size()); ++id) {
    const Eigen::Isometry3d pose_in_camera = context.T_camera_world.inverse() * poses[id];
    const double facing = (-pose_in_camera.linear().col(0)).dot(-pose_in_camera.translation());
    if (!std::isfinite(facing) || facing <= 0) continue;
    const auto projected = solver_->reproject_pose(poses[id], target_.armor_type);
    if (projected.size() != 4) continue;
    visible.push_back({id, facing, {projected[0], projected[1], projected[2], projected[3]}});
  }
  std::sort(visible.begin(), visible.end(), [](const auto & left, const auto & right) {
    return left.facing > right.facing;
  });
  if (visible.size() > 3) visible.resize(3);
  if (visible.empty()) return false;

  std::vector<Armor *> candidates;
  for (auto & armor : armors) {
    const bool finite_points = armor.points.size() == 4 &&
      std::all_of(armor.points.begin(), armor.points.end(), [](const cv::Point2f & point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      });
    if (armor.name == target_.name && armor.type == target_.armor_type && finite_points) {
      candidates.push_back(&armor);
    }
  }
  if (candidates.empty()) return false;

  const double gate = target_.matching_initialized()
                        ? estimator_config_.armor_match_gate
                        : estimator_config_.armor_match_gate_not_all_init;
  struct CandidatePair
  {
    double cost;
    int detection;
    int prediction;
  };
  std::vector<CandidatePair> pairs;
  const auto edge_angle = [](const cv::Point2f & from, const cv::Point2f & to) {
    return std::atan2(to.y - from.y, to.x - from.x);
  };
  for (int detection = 0; detection < static_cast<int>(candidates.size()); ++detection) {
    const auto & measured = candidates[detection]->points;
    for (int prediction = 0; prediction < static_cast<int>(visible.size()); ++prediction) {
      const auto & projected = visible[prediction].points;
      cv::Point2f measured_center(0, 0);
      cv::Point2f projected_center(0, 0);
      double angle_error = 0;
      double measured_perimeter = 0;
      double projected_perimeter = 0;
      for (int corner = 0; corner < 4; ++corner) {
        measured_center += measured[corner];
        projected_center += projected[corner];
        const int next = (corner + 1) % 4;
        angle_error += std::abs(tools::limit_rad(
          edge_angle(projected[corner], projected[next]) -
          edge_angle(measured[corner], measured[next])));
        measured_perimeter += cv::norm(measured[corner] - measured[next]);
        projected_perimeter += cv::norm(projected[corner] - projected[next]);
      }
      measured_center *= 0.25f;
      projected_center *= 0.25f;
      const double center_error = cv::norm(measured_center - projected_center);
      const double perimeter_error = std::abs(projected_perimeter - measured_perimeter) /
                                     std::max(projected_perimeter, 1e-6);
      const double cost = estimator_config_.armor_match_center_weight * center_error +
                          estimator_config_.armor_match_angle_weight * angle_error +
                          estimator_config_.armor_match_perimeter_weight * perimeter_error;
      if (std::isfinite(cost) && cost < gate) pairs.push_back({cost, detection, prediction});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const auto & left, const auto & right) {
    return left.cost < right.cost;
  });
  std::vector<bool> used_detections(candidates.size(), false);
  std::vector<bool> used_predictions(visible.size(), false);
  std::vector<std::pair<int, Armor>> matches;
  int primary_id = -1;
  for (const auto & pair : pairs) {
    if (used_detections[pair.detection] || used_predictions[pair.prediction]) continue;
    used_detections[pair.detection] = true;
    used_predictions[pair.prediction] = true;
    const int id = visible[pair.prediction].id;
    if (primary_id < 0) primary_id = id;
    matches.emplace_back(id, *candidates[pair.detection]);
  }
  if (matches.empty()) return false;
  if (matches.size() == 1) solver_->solve(matches.front().second);
  return target_.update(matches, primary_id, context, t) > 0;
}

}  // namespace auto_aim
