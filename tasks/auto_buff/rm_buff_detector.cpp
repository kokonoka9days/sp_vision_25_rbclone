#include "rm_buff_detector.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "vc/detector/rune_detector.h"
#include "vc/detector/rune_detector_param.h"
#include "vc/feature/rune_combo.h"
#include "vc/feature/rune_group.h"

namespace auto_buff
{

struct Rm_Buff_Detector::Impl
{
  std::unique_ptr<RuneDetector> detector;
  std::vector<FeatureNode_ptr> groups;  // 跨帧持久化的神符序列组
  int64_t tick = 0;
  PixChannel color = PixChannel::RED;
  int color_thresh = 80;
  std::optional<PowerRune> last_powerrune;

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

  std::optional<PowerRune> convertToPowerRune(cv::Mat & img)
  {
    if (groups.empty()) return std::nullopt;

    auto rune_group = RuneGroup::cast(groups.front());
    if (!rune_group) return std::nullopt;

    auto trackers = rune_group->getTrackers();
    if (trackers.empty()) return std::nullopt;

    // 收集中心点用于计算旋转中心 r_center
    std::vector<cv::Point2f> center_points;
    struct FanData
    {
      cv::Point2f center;
      std::vector<cv::Point2f> corners;
      RuneType type;
    };
    std::vector<FanData> fan_data_list;

    for (auto & tracker : trackers) {
      auto combo = RuneCombo::cast(tracker);
      if (!combo) continue;

      auto & children = combo->childFeatures();

      // 收集神符中心点
      auto center_it = children.find("center");
      if (center_it != children.end() && center_it->second) {
        center_points.push_back(center_it->second->imageCache().getCenter());
      }

      // 收集扇叶数据
      auto fan_it = children.find("fan");
      if (fan_it != children.end() && fan_it->second) {
        FanData fd;
        fd.center = fan_it->second->imageCache().getCenter();
        fd.corners = fan_it->second->imageCache().getCorners();
        fd.type = combo->getRuneType();
        fan_data_list.push_back(fd);
      }
    }

    if (fan_data_list.empty()) return std::nullopt;

    // 计算旋转中心: 所有神符中心点的均值
    cv::Point2f r_center(0, 0);
    if (!center_points.empty()) {
      for (auto & p : center_points) r_center += p;
      r_center /= static_cast<float>(center_points.size());
    } else {
      // 回退: 用扇叶中心均值
      for (auto & fd : fan_data_list) r_center += fd.center;
      r_center /= static_cast<float>(fan_data_list.size());
    }

    // 构建 FanBlade 列表
    std::vector<FanBlade> fanblades;
    for (auto & fd : fan_data_list) {
      // 确定扇叶类型
      FanBlade_type fb_type = _light;
      if (fd.type == RuneType::PENDING_STRUCK) {
        fb_type = _target;
      } else if (fd.type == RuneType::STRUCK) {
        continue;  // 跳过已击打扇叶
      }

      // 构造 6 个关键点: [4 角点, 中心, 叶尖方向]
      std::vector<cv::Point2f> kpt(6);
      for (size_t i = 0; i < 4 && i < fd.corners.size(); i++) {
        kpt[i] = fd.corners[i];
      }
      // 补齐不足 4 个角点的情况
      for (size_t i = fd.corners.size(); i < 4; i++) {
        kpt[i] = fd.center;
      }
      kpt[4] = fd.center;  // point5: 扇叶中心
      // point6: 叶尖方向 (从旋转中心向外)
      cv::Point2f dir = fd.center - r_center;
      float dir_len = cv::norm(dir);
      if (dir_len > 1e-3f) {
        kpt[5] = fd.center + dir / dir_len * 20.0f;
      } else {
        kpt[5] = fd.center;
      }

      fanblades.emplace_back(kpt, fd.center, fb_type);
    }

    if (fanblades.empty()) return std::nullopt;

    // 构建 PowerRune (传入上一帧用于抗跳变)
    // 注意: xyz_in_world / ypr_in_world / ypd_in_world 等 3D 位姿
    // 由下游 Solver::solve() 通过 PnP + 手眼标定链重新计算并覆写，
    // Rm_Buff_Detector 只负责填充 2D 图像级数据 (r_center + fanblades)
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

std::optional<PowerRune> Rm_Buff_Detector::detect(cv::Mat & img)
{
  // 构建输入
  DetectorInput input;
  input.setImage(img);
  input.setTick(impl_->tick++);
  input.setGyroData(GyroData{});
  input.setColor(impl_->color);
  input.setColorThresh(impl_->color_thresh);
  input.setFeatureNodes(impl_->groups);

  // 执行检测
  DetectorOutput output;
  impl_->detector->detect(input, output);

  // 持久化序列组状态
  impl_->groups = output.getFeatureNodes();

  if (!output.getValid() || impl_->groups.empty()) {
    return std::nullopt;
  }

  auto result = impl_->convertToPowerRune(img);
  if (result.has_value()) {
    impl_->last_powerrune = result;
  }
  return result;
}

}  // namespace auto_buff
