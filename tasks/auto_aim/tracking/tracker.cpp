#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "enemy_color_policy.hpp"
#include "../geometry/solver.hpp"
#include "tools/logger.hpp"
#include "tools/fft.hpp"
#include "tools/math_tools.hpp"


namespace auto_aim
{
namespace
{
/** @brief 计算装甲板归一化中心到图像中心的距离 @param armor 装甲板 @return 中心距离；坐标无效时返回正无穷 */
double normalized_center_distance(const Armor & armor)
{
  if (!std::isfinite(armor.center_norm.x) || !std::isfinite(armor.center_norm.y) ||
      armor.center_norm.x < 0.0F || armor.center_norm.y < 0.0F) {
    return std::numeric_limits<double>::infinity();
  }
  return cv::norm(armor.center_norm - cv::Point2f(0.5F, 0.5F));
}

/** @brief 按装甲板到图像中心的距离原地排序 @param armors 装甲板列表 */
void sort_by_image_center(std::list<Armor> & armors)
{
  armors.sort([](const Armor & a, const Armor & b) {
    return normalized_center_distance(a) < normalized_center_distance(b);
  });
}

template <typename T>
T optional_value(const YAML::Node & yaml, const char * key, const T & fallback)
{
  return yaml[key] ? yaml[key].as<T>() : fallback;
}

CenterAccelerationEstimatorConfig read_center_acceleration_config(const YAML::Node & yaml)
{
  CenterAccelerationEstimatorConfig config;
  config.enabled = optional_value(yaml, "center_accel_ff_enabled", config.enabled);
  config.window_seconds = optional_value(yaml, "center_accel_window_s", config.window_seconds);
  config.min_samples = optional_value(yaml, "center_accel_min_samples", config.min_samples);
  config.min_span_seconds =
    optional_value(yaml, "center_accel_min_span_s", config.min_span_seconds);
  config.ema_alpha = optional_value(yaml, "center_accel_ema_alpha", config.ema_alpha);
  config.max_acceleration = optional_value(yaml, "center_accel_max_mps2", config.max_acceleration);
  config.max_jerk = optional_value(yaml, "center_accel_max_jerk_mps3", config.max_jerk);
  config.max_fit_rmse = optional_value(yaml, "center_accel_max_fit_rmse_m", config.max_fit_rmse);
  config.stale_timeout_seconds =
    optional_value(yaml, "center_accel_stale_timeout_s", config.stale_timeout_seconds);
  return config;
}
}  // namespace

Tracker::Tracker(const std::string & config_path, Solver * solver)
: Tracker(config_path, static_cast<IArmorPoseSolver *>(solver))
{
}

Tracker::Tracker(const std::string & config_path, IArmorPoseSolver * solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  if (solver_ == nullptr) throw std::invalid_argument("Tracker requires a non-null Solver");
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_str_ = yaml["enemy_color"].as<std::string>();
  enemy_color_ = (enemy_color_str_ == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
  center_acceleration_estimator_ =
    CenterAccelerationEstimator(read_center_acceleration_config(yaml));

  last_cam_is_short = true;
}

void Tracker::setSolver(Solver * solver)
{
  setPoseSolver(solver);
}

void Tracker::setPoseSolver(IArmorPoseSolver * solver)
{
  if (solver == nullptr) throw std::invalid_argument("Tracker requires a non-null Solver");
  solver_ = solver;
}

std::string Tracker::state() const { return state_; }

void Tracker::reset()
{
  detect_count_ = 0;
  temp_lost_count_ = 0;
  state_ = "lost";
  pre_state_ = "lost";
  last_timestamp_ = std::chrono::steady_clock::now();
  last_mode_.reset();
  center_acceleration_estimator_.reset();
}

void Tracker::set_fft(tools::FFTExample * fft)
{
  fft_ = fft;
  reset_fft_sample_state();
}

void Tracker::reset_fft_sample_state()
{
  if (fft_) fft_->reset();
}

void Tracker::update_fft_sample(
  const Armor & armor, std::chrono::steady_clock::time_point t)
{
  if (!fft_) return;
  fft_->add_sample(t, target_.last_id, armor.xyz_in_world.z());
}

void Tracker::update_camera_mode(bool cam_is_short)
{
  if (cam_is_short != last_cam_is_short) center_acceleration_estimator_.reset();
  target_.cam_is_short = cam_is_short;
  last_cam_is_short = cam_is_short;
}

bool Tracker::use_center_acceleration() const
{
  return target_.name != ArmorName::base && target_.name != ArmorName::outpost;
}

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
  if (enemy_color_str_ == "auto") {
    enemy_color_ = enemy_color_from_gimbal(EnemyColorPolicy::Sentry, g.enemy_color);
  }

  update_camera_mode(cam_is_short);

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  filter_enemy_armors(armors, enemy_color_, use_enemy_color);

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  sort_by_image_center(armors);

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
  if (enemy_color_str_ == "auto") {
    enemy_color_ = enemy_color_from_gimbal(EnemyColorPolicy::Standard, g.enemy_color);
  }

  update_camera_mode(cam_is_short);

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  filter_enemy_armors(armors, enemy_color_, use_enemy_color);

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  sort_by_image_center(armors);

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  // armors.sort(
  //   [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found = 0;

  if (!last_mode_.has_value()) last_mode_ = g.mode;
  bool mode_switch_0to1 = (*last_mode_ == 0 && g.mode == 1);
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
  last_mode_ = g.mode;
  // found = set_target(armors, t);

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    // tools::logger()->debug("[Tracker] Target diverged!");
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


std::list<Target> Tracker::test_track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool cam_is_short, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;
  
 
  update_camera_mode(cam_is_short);

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  filter_enemy_armors(armors, enemy_color_, use_enemy_color);

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  sort_by_image_center(armors);

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

  filter_enemy_armors(armors, enemy_color_, use_enemy_color);

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  sort_by_image_center(armors);

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

  const bool cam_is_short = target_.cam_is_short;
  auto & armor = armors.front();
  if (!solver_->try_solve(armor)) return false;

  // 根据兵种优化初始化参数
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

  target_.cam_is_short = cam_is_short;
  last_cam_is_short = cam_is_short;

  center_acceleration_estimator_.reset();
  if (use_center_acceleration()) {
    const Eigen::VectorXd state = target_.ekf_x();
    center_acceleration_estimator_.add_sample(t, {state[0], state[2]});
  }

  reset_fft_sample_state();
  update_fft_sample(armor, t);
  return true;
}
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if(fft_ != nullptr ){
    auto wave = fft_->get_wave();
    target_.wave_ = wave.valid()
              ? std::make_optional(std::move(wave))
              : std::nullopt;

  }
  Eigen::VectorXd acceleration = Eigen::VectorXd::Zero(3);
  if (use_center_acceleration()) {
    acceleration.head<2>() = center_acceleration_estimator_.acceleration(t);
  }
  // XY acceleration is intentionally limited to this online frame-to-frame prediction.
  target_.predict(t, acceleration);

  bool found = false;

  // 由于 armors 在 track/sb_track 中已经按距离图像中心的远近排序
  // 遍历找到的第一个匹配目标的装甲板，即为视野中最居中、畸变最小的装甲板
  for (auto & armor : armors) {
    if (armor.name == target_.name && armor.type == target_.armor_type) {
      if (!solver_->try_solve(armor)) continue;
      // update 返回 false 说明观测未匹配到装甲板、EKF 未执行校正，滤波器状态没有变化，
      // 不能算作跟踪成功，继续尝试后续装甲板
      if (!target_.update(armor)) continue;

      if (use_center_acceleration()) {
        const Eigen::VectorXd state = target_.ekf_x();
        center_acceleration_estimator_.add_sample(t, {state[0], state[2]});
      }

      update_fft_sample(armor, t);
      found = true;
      break; // 找到最优匹配后立即退出
    }
  }

  return found;
}

}  // namespace auto_aim
