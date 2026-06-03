#include "rm_buff_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
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

struct Rm_Buff_Detector::Impl
{
  std::unique_ptr<RuneDetector> detector;
  std::vector<FeatureNode_ptr> groups;
  int64_t tick = 0;
  PixChannel color = PixChannel::RED;
  int color_thresh = 80;
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
      if (cfg["color"]) {
        int c = cfg["color"].as<int>();
        color = (c == 1) ? PixChannel::BLUE : PixChannel::RED;
      }
      if (cfg["color_thresh"]) {
        color_thresh = cfg["color_thresh"].as<int>();
      }
      tools::logger()->info(
        "[Rm_Buff_Detector] 配置加载完成 color={} thresh={}",
        static_cast<int>(color), color_thresh);
    } catch (const std::exception & e) {
      tools::logger()->warn(
        "[Rm_Buff_Detector] 配置加载失败: {}, 使用默认参数", e.what());
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
      cv::Point2f fan_center;      // 悬臂中心 (用于计算 kpt[5])
      std::vector<cv::Point2f> target_corners;  // 靶标菱形四角 (kpt[0..3])
      RuneType type;
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

      // 悬臂中心: 来自 RUNE_FAN (用于 kpt[5] 计算)
      cv::Point2f fan_center(0, 0);
      auto fan_it = children.find(FeatureNode::ChildFeatureType::RUNE_FAN);
      if (fan_it != children.end() && fan_it->second) {
        fan_center = fan_it->second->imageCache().getCenter();
      }

      // ★ 靶标角点: 来自 RuneCombo 自身的 imageCache (150mm 菱形)
      auto & combo_cache = combo->imageCache();
      auto combo_corners = combo_cache.getCorners();
      if (combo_corners.size() >= 4) {
        FanData fd;
        fd.target_corners = combo_corners;
        fd.target_center = combo_cache.getCenter();
        fd.fan_center = fan_center;
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

    // ★ 关键: 在绘制和构建 FanBlade 之前统一排序所有靶标角点
    // RuneCombo 的 corners 顺序为 {top, right, bottom, left}
    // 需要重排为 Solver 期望的 {bottom, right, top, left} (0=远,2=近)
    for (auto & fd : fan_data_list) {
      if (fd.target_corners.size() >= 4) sortCorners(fd.target_corners, r_center);
    }

    // ====== 调试绘制 (靶标菱形角点+中心+方向, 编号对应 kpt[0..5]) ======
    if (debug_draw) {
      for (auto & fd : fan_data_list) {
        if (fd.type == RuneType::STRUCK) continue;
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

        // 点 6: kpt[5] = 从悬臂中心指向 R 方向
        cv::Point2f dir = r_center - fd.fan_center;
        float dir_len = cv::norm(dir);
        cv::Point2f pt6 = (dir_len > 1e-3f) ? fd.fan_center + dir / dir_len * 30.0f
                                            : fd.fan_center;
        cv::circle(img, pt6, 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(
          img, "6", pt6 + cv::Point2f(6, -6),
          cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1);
      }
      // 旋转中心
      cv::drawMarker(
        img, r_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
      cv::putText(
        img, "R", r_center + cv::Point2f(12, -8),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      cv::putText(
        img, "rm_vision:" + std::to_string(tick), cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
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

      kpt[4] = fd.target_center;  // point5: 扇叶中心
      // 6号点(kpt[5]): 连接处顶部, 从扇叶中心指向旋转中心方向
      cv::Point2f dir = r_center - fd.fan_center;  // 指向 R
      float dir_len = cv::norm(dir);
      kpt[5] = (dir_len > 1e-3f) ? fd.fan_center + dir / dir_len * 30.0f
                                 : fd.fan_center;  // point6: 叶尖方向

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

std::optional<PowerRune> Rm_Buff_Detector::detect(cv::Mat & img)
{
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
