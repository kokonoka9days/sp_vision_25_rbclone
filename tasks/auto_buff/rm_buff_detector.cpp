#include "rm_buff_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fmt/core.h>

#include "tools/logger.hpp"
#include "vc/detector/rune_detector.h"
#include "vc/detector/rune_detector_param.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_group.h"
#include "vc/feature/feature_node_child_feature_type.h"
#include "vc/feature/tracking_feature_node.h"

namespace auto_buff
{
namespace
{
constexpr double SMALL_BUFF_ANGULAR_SPEED = 1.0471975511965976;  // pi / 3 rad/s
constexpr double MIN_DIRECTION_SPEED = 0.05;
}  // namespace

struct Rm_Buff_Detector::Impl
{
  std::unique_ptr<RuneDetector> detector;
  std::vector<FeatureNode_ptr> groups;
  int64_t tick = 0;
  PixChannel color = PixChannel::BLUE;
  bool auto_enemy_color = false;
  int color_thresh = 80;
  bool auto_color_thresh = true;
  int color_thresh_min = 35;
  int color_thresh_max = 140;
  int color_thresh_max_step = 4;
  double color_thresh_alpha = 0.2;
  double min_foreground_ratio = 0.0001;
  double max_foreground_ratio = 0.15;
  bool auto_thresh_use_roi = true;
  double auto_thresh_roi_scale = 1.35;
  int auto_thresh_roi_min_size = 240;
  cv::Rect last_threshold_roi;
  bool threshold_roi_from_tracking = false;
  double prediction_box_time = 0.08;
  double prediction_velocity_alpha = 0.25;
  double prediction_max_angular_speed = 3.0;
  std::chrono::steady_clock::time_point frame_timestamp = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_target_timestamp;
  std::optional<double> last_target_angle;
  double filtered_measured_angular_velocity = 0.0;
  double predicted_angular_velocity = 0.0;
  bool prediction_velocity_ready = false;
  std::optional<PowerRune> last_powerrune;
  bool debug_draw = false;

  explicit Impl(const std::string & config_path)
  {
    detector = RuneDetector::make_detector();
    loadConfig(config_path);
  }

  void loadConfig(const std::string & config_path)
  {
    try {
      YAML::Node cfg = YAML::LoadFile(config_path);
      if (cfg["enemy_color"]) {
        const auto enemy_color = cfg["enemy_color"].as<std::string>();
        if (enemy_color == "auto") {
          auto_enemy_color = true;
        } else if (enemy_color == "red") {
          color = PixChannel::RED;
        } else if (enemy_color == "blue") {
          color = PixChannel::BLUE;
        } else {
          tools::logger()->warn(
            "[Rm_Buff_Detector] 未知 enemy_color={}，默认使用蓝色", enemy_color);
        }
      } else if (cfg["color"]) {
        // 兼容旧版 rm_buff_config.yaml: 0 = 红色，1 = 蓝色。
        int c = cfg["color"].as<int>();
        color = (c == 1) ? PixChannel::BLUE : PixChannel::RED;
      }
      if (cfg["color_thresh"]) {
        color_thresh = cfg["color_thresh"].as<int>();
      }
      if (cfg["auto_color_thresh"]) {
        auto_color_thresh = cfg["auto_color_thresh"].as<bool>();
      }
      if (cfg["color_thresh_min"]) {
        color_thresh_min = cfg["color_thresh_min"].as<int>();
      }
      if (cfg["color_thresh_max"]) {
        color_thresh_max = cfg["color_thresh_max"].as<int>();
      }
      if (cfg["color_thresh_max_step"]) {
        color_thresh_max_step = cfg["color_thresh_max_step"].as<int>();
      }
      if (cfg["color_thresh_alpha"]) {
        color_thresh_alpha = cfg["color_thresh_alpha"].as<double>();
      }
      if (cfg["min_foreground_ratio"]) {
        min_foreground_ratio = cfg["min_foreground_ratio"].as<double>();
      }
      if (cfg["max_foreground_ratio"]) {
        max_foreground_ratio = cfg["max_foreground_ratio"].as<double>();
      }
      if (cfg["auto_thresh_use_roi"]) {
        auto_thresh_use_roi = cfg["auto_thresh_use_roi"].as<bool>();
      }
      if (cfg["auto_thresh_roi_scale"]) {
        auto_thresh_roi_scale = cfg["auto_thresh_roi_scale"].as<double>();
      }
      if (cfg["auto_thresh_roi_min_size"]) {
        auto_thresh_roi_min_size = cfg["auto_thresh_roi_min_size"].as<int>();
      }
      if (cfg["prediction_box_time"]) {
        prediction_box_time = cfg["prediction_box_time"].as<double>();
      }
      if (cfg["prediction_velocity_alpha"]) {
        prediction_velocity_alpha = cfg["prediction_velocity_alpha"].as<double>();
      }
      if (cfg["prediction_max_angular_speed"]) {
        prediction_max_angular_speed = cfg["prediction_max_angular_speed"].as<double>();
      }

      color_thresh_min = std::clamp(color_thresh_min, 0, 255);
      color_thresh_max = std::clamp(color_thresh_max, color_thresh_min, 255);
      color_thresh = std::clamp(color_thresh, color_thresh_min, color_thresh_max);
      color_thresh_max_step = std::max(color_thresh_max_step, 1);
      color_thresh_alpha = std::clamp(color_thresh_alpha, 0.0, 1.0);
      min_foreground_ratio = std::clamp(min_foreground_ratio, 0.0, 1.0);
      max_foreground_ratio =
        std::clamp(max_foreground_ratio, min_foreground_ratio, 1.0);
      auto_thresh_roi_scale = std::max(auto_thresh_roi_scale, 1.0);
      auto_thresh_roi_min_size = std::max(auto_thresh_roi_min_size, 32);
      prediction_box_time = std::clamp(prediction_box_time, 0.0, 0.5);
      prediction_velocity_alpha = std::clamp(prediction_velocity_alpha, 0.0, 1.0);
      prediction_max_angular_speed = std::max(prediction_max_angular_speed, 0.1);

      tools::logger()->info(
        "[Rm_Buff_Detector] 配置加载完成 color={} auto_enemy_color={} thresh={} auto_thresh={}",
        static_cast<int>(color), auto_enemy_color, color_thresh, auto_color_thresh);
    } catch (const std::exception & e) {
      tools::logger()->warn(
        "[Rm_Buff_Detector] 配置加载失败: {}, 使用默认参数", e.what());
    }
  }

  std::optional<cv::Rect> getThresholdRoi(const cv::Mat & img) const
  {
    if (!auto_thresh_use_roi || groups.empty()) return std::nullopt;

    auto rune_group = RuneGroup::cast(groups.front());
    if (!rune_group) return std::nullopt;

    std::vector<cv::Point2f> points;
    for (const auto & tracker : rune_group->getTrackers()) {
      auto tracking = TrackingFeatureNode::cast(tracker);
      if (!tracking || tracking->getHistoryNodes().empty()) continue;

      auto combo = RuneCombo::cast(tracking->getHistoryNodes().front());
      if (!combo) continue;

      const auto & cache = combo->imageCache();
      points.push_back(cache.getCenter());
      const auto corners = cache.getCorners();
      points.insert(points.end(), corners.begin(), corners.end());
    }
    if (points.empty()) return std::nullopt;

    const cv::Rect2f bounds = cv::boundingRect(points);
    const cv::Point2f center(
      bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f);
    const double side = std::max(
      static_cast<double>(auto_thresh_roi_min_size),
      std::max(bounds.width, bounds.height) * auto_thresh_roi_scale);

    cv::Rect roi(
      static_cast<int>(std::floor(center.x - side * 0.5)),
      static_cast<int>(std::floor(center.y - side * 0.5)),
      static_cast<int>(std::ceil(side)), static_cast<int>(std::ceil(side)));
    roi &= cv::Rect(0, 0, img.cols, img.rows);
    if (roi.width < 32 || roi.height < 32) return std::nullopt;
    return roi;
  }

  void updateColorThreshold(const cv::Mat & img)
  {
    if (!auto_color_thresh || img.empty() || img.type() != CV_8UC3) return;

    const auto tracked_roi = getThresholdRoi(img);
    last_threshold_roi = tracked_roi.value_or(cv::Rect(0, 0, img.cols, img.rows));
    threshold_roi_from_tracking = tracked_roi.has_value();
    const cv::Mat threshold_image = img(last_threshold_roi);

    cv::Mat target_channel;
    cv::Mat opposite_channel;
    cv::extractChannel(threshold_image, target_channel, static_cast<int>(color));
    cv::extractChannel(
      threshold_image, opposite_channel,
      static_cast<int>(color == PixChannel::RED ? PixChannel::BLUE : PixChannel::RED));

    cv::Mat color_difference;
    cv::subtract(target_channel, opposite_channel, color_difference);

    cv::Mat sampled_difference;
    constexpr int SAMPLE_WIDTH = 480;
    if (color_difference.cols > SAMPLE_WIDTH) {
      const double scale = static_cast<double>(SAMPLE_WIDTH) / color_difference.cols;
      cv::resize(
        color_difference, sampled_difference, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
      sampled_difference = color_difference;
    }

    cv::Mat otsu_binary;
    const int otsu_thresh = static_cast<int>(std::lround(cv::threshold(
      sampled_difference, otsu_binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU)));
    const int candidate = std::clamp(otsu_thresh, color_thresh_min, color_thresh_max);

    cv::Mat candidate_binary;
    cv::compare(sampled_difference, candidate, candidate_binary, cv::CMP_GT);
    const double foreground_ratio =
      static_cast<double>(cv::countNonZero(candidate_binary)) / candidate_binary.total();
    if (foreground_ratio < min_foreground_ratio || foreground_ratio > max_foreground_ratio) {
      return;
    }

    const int smoothed = static_cast<int>(
      std::lround((1.0 - color_thresh_alpha) * color_thresh + color_thresh_alpha * candidate));
    const int delta = std::clamp(
      smoothed - color_thresh, -color_thresh_max_step, color_thresh_max_step);
    color_thresh = std::clamp(color_thresh + delta, color_thresh_min, color_thresh_max);

    if (tick % 60 == 0) {
      tools::logger()->debug(
        "[Rm_Buff_Detector] 自动阈值={} otsu={} foreground={:.4f} roi={}x{} tracked={}",
        color_thresh, otsu_thresh, foreground_ratio, last_threshold_roi.width,
        last_threshold_roi.height, threshold_roi_from_tracking);
    }
  }

  /**
   * @brief 将 4 个角点排序为 Solver::solvePnP 期望的菱形顺序
   *
   * YOLO 6 点布局 (用户描述):
   *   1号点=下边缘, 2号点=右边缘, 3号点=上边缘, 4号点=左边缘
   *   5号点=靶心中心, 6号点=连接处顶部(最靠近旋转中心)
   *
   *   kpt[0] = 1号点 下边缘 (最远离旋转中心)
   *   kpt[1] = 2号点 右边缘
   *   kpt[2] = 3号点 上边缘 (最靠近旋转中心)
   *   kpt[3] = 4号点 左边缘
   *   kpt[4] = 5号点 靶心中心
   *   kpt[5] = 6号点 连接处顶部
   *
   * Solver 使用 kpt[0..3] 做 PnP, 对应 3D 模型:
   *   [0] → (0, 0, 827e-3)       下边缘 (最远离旋转中心)
   *   [1] → (0,  127e-3, 700e-3) 右边缘
   *   [2] → (0, 0, 573e-3)       上边缘 (最靠近旋转中心)
   *   [3] → (0, -127e-3, 700e-3) 左边缘
   *
   * rm_vision_core 的 fan corners 是扇叶轮廓四角, 与上述物理点大致对应,
   * 但顺序随机。此函数按几何关系重新排序。
   */
  static void sortCorners(
    std::vector<cv::Point2f> & corners, const cv::Point2f & r_center)
  {
    if (corners.size() < 4) return;

    // 按到旋转中心的距离: 最远↔下边缘(kpt[0]), 最近↔上边缘(kpt[2])
    std::vector<std::pair<float, int>> dists;
    for (int i = 0; i < 4; i++)
      dists.emplace_back(cv::norm(corners[i] - r_center), i);
    std::sort(dists.begin(), dists.end(),
              [](auto & a, auto & b) { return a.first < b.first; });

    cv::Point2f sorted[4];
    sorted[0] = corners[dists[3].second];  // 最远 → kpt[0] 下边缘
    sorted[2] = corners[dists[0].second];  // 最近 → kpt[2] 上边缘

    // 剩余两个用叉积分左右: ref = 下→上方向, cross>0 → 右侧
    cv::Point2f mid1 = corners[dists[1].second];
    cv::Point2f mid2 = corners[dists[2].second];
    cv::Point2f ref = sorted[2] - sorted[0];
    cv::Point2f to_mid1 = mid1 - sorted[0];
    float cross = ref.x * to_mid1.y - ref.y * to_mid1.x;
    if (cross > 0) {
      sorted[1] = mid1;  // kpt[1] 右角
      sorted[3] = mid2;  // kpt[3] 左角
    } else {
      sorted[1] = mid2;  // kpt[1] 右角
      sorted[3] = mid1;
    }
    for (int i = 0; i < 4; i++) corners[i] = sorted[i];
  }

  std::optional<PowerRune> convertToPowerRune(cv::Mat & img)
  {
    if (groups.empty()) { fmt::print("[rm_cvt] groups empty\n"); return std::nullopt; }

    auto rune_group = RuneGroup::cast(groups.front());
    if (!rune_group) { fmt::print("[rm_cvt] rune_group cast failed\n"); return std::nullopt; }

    auto trackers = rune_group->getTrackers();
    if (trackers.empty()) { fmt::print("[rm_cvt] trackers empty\n"); return std::nullopt; }

    // 收集中心点用于计算旋转中心 r_center
    std::vector<cv::Point2f> center_points;
    struct FanData
    {
      cv::Point2f target_center;   // 靶标中心 (kpt[4])
      cv::Point2f fan_inner_mid;   // 悬臂内侧边中点 (kpt[5])
      std::vector<cv::Point2f> target_corners;  // 靶标菱形四角 (kpt[0..3])
      std::vector<cv::Point2f> fan_corners;     // 悬臂矩形四角 (调试绘制)
      RuneType type;
      int fan_inner_i1 = -1;  // 内侧边两个角点索引 (距 R 中心最近)
      int fan_inner_i2 = -1;
    };
    std::vector<FanData> fan_data_list;

    for (auto & tracker : trackers) {
      auto tracking = TrackingFeatureNode::cast(tracker);
      if (!tracking) continue;
      auto & history = tracking->getHistoryNodes();
      if (history.empty()) continue;
      auto combo = RuneCombo::cast(history.front());
      if (!combo) continue;

      // 旋转中心: 来自 RUNE_CENTER 子特征
      auto & children = combo->childFeatures();
      auto center_it = children.find(FeatureNode::ChildFeatureType::RUNE_CENTER);
      if (center_it != children.end() && center_it->second) {
        center_points.push_back(center_it->second->imageCache().getCenter());
      }

      // 悬臂: 来自 RUNE_FAN, 仅收集四角, 内侧边中点稍后用精确 r_center 计算
      std::vector<cv::Point2f> fan_corners;
      auto fan_it = children.find(FeatureNode::ChildFeatureType::RUNE_FAN);
      if (fan_it != children.end() && fan_it->second) {
        fan_corners = fan_it->second->imageCache().getCorners();
      }

      // 靶标角点: 来自 RuneCombo 自身的 imageCache (150mm 菱形)
      auto & combo_cache = combo->imageCache();
      auto combo_corners = combo_cache.getCorners();
      if (combo_corners.size() >= 4) {
        FanData fd;
        fd.target_corners = combo_corners;
        fd.target_center = combo_cache.getCenter();
        fd.fan_corners = fan_corners;
        fd.fan_inner_mid = cv::Point2f(0, 0);  // 稍后用精确 r_center 计算
        fd.type = combo->getRuneType();
        fan_data_list.push_back(fd);
      }
    }

    if (fan_data_list.empty()) {
      fmt::print("[rm_cvt] fan_data_list empty (trackers={})\n", trackers.size());
      return std::nullopt;
    }
    static int cvt_ok_count = 0;
    if (++cvt_ok_count % 20 == 1)
      fmt::print("[rm_cvt] OK trackers={} fans={}\n", trackers.size(), fan_data_list.size());

    // 计算旋转中心
    cv::Point2f r_center(0, 0);
    if (!center_points.empty()) {
      for (auto & p : center_points) r_center += p;
      r_center /= static_cast<float>(center_points.size());
    } else {
      for (auto & fd : fan_data_list) r_center += fd.target_center;
      r_center /= static_cast<float>(fan_data_list.size());
    }

    // 使用精确 r_center 计算每个扇叶的悬臂内侧边中点 = kpt[5]
    for (auto & fd : fan_data_list) {
      if (fd.fan_corners.size() >= 4 && cv::norm(r_center) > 0) {
        std::vector<std::pair<float, int>> dists;
        for (size_t ci = 0; ci < fd.fan_corners.size(); ci++)
          dists.emplace_back(cv::norm(fd.fan_corners[ci] - r_center), ci);
        std::sort(dists.begin(), dists.end(),
                  [](auto & a, auto & b) { return a.first < b.first; });
        fd.fan_inner_i1 = dists[0].second;
        fd.fan_inner_i2 = dists[1].second;
        fd.fan_inner_mid = (fd.fan_corners[fd.fan_inner_i1] +
                            fd.fan_corners[fd.fan_inner_i2]) * 0.5f;
      } else if (!fd.fan_corners.empty()) {
        // 退路: 角点不足 4 个时用靶标中心近似
        fd.fan_inner_mid = fd.target_center;
      }
    }

    // ★ 关键: 在绘制和构建 FanBlade 之前统一排序所有靶标角点
    // RuneCombo 的 corners 顺序为 {top, right, bottom, left}
    // 需要重排为 Solver 期望的 {bottom, right, top, left} (0=远,2=近)
    for (auto & fd : fan_data_list) {
      if (fd.target_corners.size() >= 4) sortCorners(fd.target_corners, r_center);
    }

    std::vector<cv::Point2f> predicted_target_corners;
    cv::Point2f predicted_target_center;
    double prediction_forward_time = 0.0;
    auto target_it = std::find_if(
      fan_data_list.begin(), fan_data_list.end(),
      [](const FanData & fd) { return fd.type == RuneType::PENDING_STRUCK; });
    const bool prediction_target_found = target_it != fan_data_list.end();
    if (target_it != fan_data_list.end()) {
      const auto target_vector = target_it->target_center - r_center;
      const double target_angle = std::atan2(target_vector.y, target_vector.x);

      if (last_target_angle.has_value()) {
        const double dt = std::chrono::duration<double>(
          frame_timestamp - last_target_timestamp).count();
        const double angle_delta = std::atan2(
          std::sin(target_angle - last_target_angle.value()),
          std::cos(target_angle - last_target_angle.value()));
        if (dt > 1e-4 && dt < 0.25) {
          const double measured_velocity = angle_delta / dt;
          if (std::abs(measured_velocity) <= prediction_max_angular_speed) {
            if (!prediction_velocity_ready) {
              filtered_measured_angular_velocity = measured_velocity;
              if (std::abs(filtered_measured_angular_velocity) >= MIN_DIRECTION_SPEED) {
                predicted_angular_velocity = std::copysign(
                  SMALL_BUFF_ANGULAR_SPEED, filtered_measured_angular_velocity);
                prediction_velocity_ready = true;
                tools::logger()->info(
                  "[Rm_Buff_Detector] 小符预测框已启用 direction={} speed={:.3f}rad/s "
                  "measured={:.3f}rad/s",
                  predicted_angular_velocity > 0.0 ? "CW" : "CCW",
                  std::abs(predicted_angular_velocity), measured_velocity);
              }
            } else {
              filtered_measured_angular_velocity =
                (1.0 - prediction_velocity_alpha) * filtered_measured_angular_velocity +
                prediction_velocity_alpha * measured_velocity;
              if (std::abs(filtered_measured_angular_velocity) >= MIN_DIRECTION_SPEED) {
                predicted_angular_velocity = std::copysign(
                  SMALL_BUFF_ANGULAR_SPEED, filtered_measured_angular_velocity);
              }
            }
          }
        }
      }
      last_target_angle = target_angle;
      last_target_timestamp = frame_timestamp;

      if (prediction_velocity_ready) {
        const double processing_delay = std::clamp(
          std::chrono::duration<double>(
            std::chrono::steady_clock::now() - frame_timestamp).count(),
          0.0, 0.2);
        prediction_forward_time = prediction_box_time + processing_delay;
        const double prediction_angle = predicted_angular_velocity * prediction_forward_time;
        const double cos_angle = std::cos(prediction_angle);
        const double sin_angle = std::sin(prediction_angle);
        auto rotate_point = [&](const cv::Point2f & point) {
          const auto offset = point - r_center;
          return r_center + cv::Point2f(
            static_cast<float>(offset.x * cos_angle - offset.y * sin_angle),
            static_cast<float>(offset.x * sin_angle + offset.y * cos_angle));
        };

        predicted_target_corners.reserve(target_it->target_corners.size());
        for (const auto & point : target_it->target_corners) {
          predicted_target_corners.push_back(rotate_point(point));
        }
        predicted_target_center = rotate_point(target_it->target_center);
      }
    }

    // ====== 调试绘制 (靶标菱形角点+中心+方向, 编号对应 kpt[0..5]) ======
    if (debug_draw) {
      for (auto & fd : fan_data_list) {
        if (fd.type == RuneType::STRUCK) {
          // 已打击: 深色绘制 (菱形 + 悬臂矩形 + 内侧边高亮)
          if (fd.target_corners.size() >= 4) {
            for (size_t i = 0; i < 4; i++) {
              cv::line(
                img, fd.target_corners[i], fd.target_corners[(i + 1) % 4],
                cv::Scalar(100, 100, 100), 1);
            }
          }
          if (fd.fan_corners.size() >= 4) {
            for (size_t i = 0; i < fd.fan_corners.size(); i++) {
              cv::line(
                img, fd.fan_corners[i], fd.fan_corners[(i + 1) % fd.fan_corners.size()],
                cv::Scalar(80, 80, 80), 1);
            }
            // 内侧边高亮 (暗红)
            if (fd.fan_inner_i1 >= 0 && fd.fan_inner_i2 >= 0) {
              cv::line(img, fd.fan_corners[fd.fan_inner_i1],
                       fd.fan_corners[fd.fan_inner_i2],
                       cv::Scalar(60, 60, 120), 2);
            }
          }
          cv::circle(img, fd.target_center, 3, cv::Scalar(100, 100, 100), -1);
          cv::circle(img, fd.fan_inner_mid, 3, cv::Scalar(100, 100, 100), -1);
          continue;
        }
        bool is_target = (fd.type == RuneType::PENDING_STRUCK);
        cv::Scalar draw_color =
          is_target ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 0, 255);

        if (fd.target_corners.size() >= 4) {
          // 菱形轮廓 (靶标)
          for (size_t i = 0; i < 4; i++) {
            cv::line(
              img, fd.target_corners[i], fd.target_corners[(i + 1) % 4],
              is_target ? cv::Scalar(0, 255, 255) : cv::Scalar(200, 200, 200),
              is_target ? 2 : 1);
          }
          // 角点 1~4: kpt[0..3]
          const char * corner_labels[4] = {"1", "2", "3", "4"};
          for (int i = 0; i < 4; i++) {
            cv::circle(img, fd.target_corners[i], 4, draw_color, -1);
            cv::putText(
              img, corner_labels[i], fd.target_corners[i] + cv::Point2f(6, -6),
              cv::FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1);
          }
        }
        // 点 5: kpt[4] = 靶标中心
        cv::circle(img, fd.target_center, 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(
          img, "5", fd.target_center + cv::Point2f(8, -8),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);

        // 悬臂矩形轮廓 (灰色)
        if (fd.fan_corners.size() >= 4) {
          for (size_t i = 0; i < fd.fan_corners.size(); i++) {
            cv::line(
              img, fd.fan_corners[i], fd.fan_corners[(i + 1) % fd.fan_corners.size()],
              cv::Scalar(150, 150, 150), 1);
          }
          // 内侧边 (靠近 R) 高亮 — 使用精确索引
          if (fd.fan_inner_i1 >= 0 && fd.fan_inner_i2 >= 0) {
            cv::line(img, fd.fan_corners[fd.fan_inner_i1],
                     fd.fan_corners[fd.fan_inner_i2], cv::Scalar(0, 0, 255), 2);
          }
        }
        // 点 6: kpt[5] = 悬臂内侧边中点 (靠近 R 标)
        cv::circle(img, fd.fan_inner_mid, 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(
          img, "6", fd.fan_inner_mid + cv::Point2f(6, -6),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1);
      }

      if (predicted_target_corners.size() >= 4) {
        const cv::Scalar prediction_color(0, 255, 0);
        for (size_t i = 0; i < predicted_target_corners.size(); ++i) {
          cv::line(
            img, predicted_target_corners[i],
            predicted_target_corners[(i + 1) % predicted_target_corners.size()],
            prediction_color, 3);
        }
        cv::circle(img, predicted_target_center, 6, prediction_color, 3);
        cv::putText(
          img, fmt::format("PRED +{:.0f}ms", prediction_forward_time * 1000.0),
          predicted_target_center + cv::Point2f(8, -8),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, prediction_color, 2);
      }

      if (threshold_roi_from_tracking && last_threshold_roi.area() > 0) {
        const cv::Scalar roi_color(255, 255, 0);
        cv::rectangle(img, last_threshold_roi, roi_color, 1);
        cv::putText(
          img, "THRESH ROI", last_threshold_roi.tl() + cv::Point(4, 18),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1);
      }

      // 旋转中心
      cv::drawMarker(
        img, r_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
      cv::putText(
        img, "R", r_center + cv::Point2f(12, -8),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      cv::putText(
        img, "rm_vision:" + std::to_string(tick) + " thresh:" + std::to_string(color_thresh),
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
      const std::string prediction_status = !prediction_target_found
                                              ? "PRED: NO TARGET"
                                              : prediction_velocity_ready
                                                  ? fmt::format(
                                                      "PRED w={:.2f}rad/s",
                                                      predicted_angular_velocity)
                                                  : "PRED: WAIT MOTION";
      cv::putText(
        img, prediction_status, cv::Point(10, 58), cv::FONT_HERSHEY_SIMPLEX, 0.65,
        prediction_velocity_ready ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 2);
    }

    // 构建 FanBlade 列表 — 使用靶标(target)角点, 确保 _target 在 fanblades[0]
    std::vector<FanBlade> fanblades;
    std::vector<FanBlade> light_blades;
    bool print_types = (cvt_ok_count % 20 == 1);
    for (size_t fi = 0; fi < fan_data_list.size(); fi++) {
      auto & fd = fan_data_list[fi];
      FanBlade_type fb_type = _light;
      if (fd.type == RuneType::PENDING_STRUCK) {
        fb_type = _target;
      } else if (fd.type == RuneType::STRUCK) {
        continue;
      }

      std::vector<cv::Point2f> kpt(6);
      if (fd.target_corners.size() >= 4) {
        for (int i = 0; i < 4; i++) kpt[i] = fd.target_corners[i];
      } else {
        for (size_t i = 0; i < fd.target_corners.size(); i++) kpt[i] = fd.target_corners[i];
      }
      for (size_t i = fd.target_corners.size(); i < 4; i++) kpt[i] = fd.target_center;

      kpt[4] = fd.target_center;  // point5: 靶标中心
      kpt[5] = fd.fan_inner_mid;  // point6: 悬臂内侧边中点

      FanBlade blade(kpt, fd.target_center, fb_type);
      if (fb_type == _target) {
        fanblades.insert(fanblades.begin(), std::move(blade));
      } else {
        light_blades.push_back(std::move(blade));
      }
    }
    fanblades.insert(fanblades.end(), light_blades.begin(), light_blades.end());

    if (fanblades.empty()) return std::nullopt;

    // 诊断: 打印最终 fanblades 顺序 (index 0 = target)
    if (print_types) {
      for (size_t bi = 0; bi < fanblades.size(); bi++) {
        fmt::print("[rm_cvt]   fanblades[{}] type={} center=({:.0f},{:.0f})\n",
                   bi, (fanblades[bi].type == _target ? "TARGET" : "light"),
                   fanblades[bi].center.x, fanblades[bi].center.y);
      }
    }

    PowerRune powerrune(fanblades, r_center, last_powerrune);
    if (powerrune.is_unsolve()) return std::nullopt;

    return powerrune;
  }
};

Rm_Buff_Detector::Rm_Buff_Detector(const std::string & config_path)
: impl_(std::make_unique<Impl>(config_path))
{
}

Rm_Buff_Detector::~Rm_Buff_Detector() = default;

void Rm_Buff_Detector::set_debug_draw(bool enable) { impl_->debug_draw = enable; }

bool Rm_Buff_Detector::is_debug_draw() const { return impl_->debug_draw; }

void Rm_Buff_Detector::set_enemy_color(uint8_t enemy_color)
{
  if (!impl_->auto_enemy_color) return;

  const auto color = enemy_color == 0 ? PixChannel::BLUE : PixChannel::RED;
  if (color == impl_->color) return;

  impl_->color = color;
  impl_->groups.clear();
  impl_->last_powerrune.reset();
  impl_->last_target_angle.reset();
  impl_->filtered_measured_angular_velocity = 0.0;
  impl_->prediction_velocity_ready = false;
  tools::logger()->info(
    "[Rm_Buff_Detector] 自动切换敌方颜色为{}", enemy_color == 0 ? "蓝色" : "红色");
}

std::optional<PowerRune> Rm_Buff_Detector::detect(cv::Mat & img)
{
  return detect(img, std::chrono::steady_clock::now());
}

std::optional<PowerRune> Rm_Buff_Detector::detect(
  cv::Mat & img, std::chrono::steady_clock::time_point timestamp)
{
  impl_->frame_timestamp = timestamp;
  impl_->updateColorThreshold(img);

  DetectorInput input;
  input.setImage(img);
  input.setTick(impl_->tick++);
  input.setGyroData(GyroData{});
  input.setColor(impl_->color);
  input.setColorThresh(impl_->color_thresh);
  input.setFeatureNodes(impl_->groups);

  DetectorOutput output;
  impl_->detector->detect(input, output);

  impl_->groups = output.getFeatureNodes();

  if (!output.getValid() || impl_->groups.empty()) {
    static int fail_count = 0;
    if (++fail_count % 30 == 1)
      fmt::print("[rm_det] frame {}: RuneDetector fail valid={} groups_empty={}\n",
                 impl_->tick, output.getValid(), impl_->groups.empty());
    return std::nullopt; 
  }

  auto result = impl_->convertToPowerRune(img);
  if (result.has_value()) {
    impl_->last_powerrune = result;
  }
  return result;
}

}  // namespace auto_buff
