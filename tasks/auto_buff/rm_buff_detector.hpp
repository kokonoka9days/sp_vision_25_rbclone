#ifndef AUTO_BUFF__RM_BUFF_DETECTOR_HPP
#define AUTO_BUFF__RM_BUFF_DETECTOR_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "buff_type.hpp"

namespace auto_buff
{

/**
 * @brief 基于 rm_vision_core (经典 CV 特征识别) 的神符检测器
 *
 * 与 Buff_Detector (YOLO) 并行，输出统一的 std::optional<PowerRune>，
 * 下游处理逻辑保持不变。
 *
 * 用法:
 *   auto_buff::Rm_Buff_Detector detector(config_path);
 *   auto power_runes = detector.detect(img);
 */
class Rm_Buff_Detector
{
public:
  /**
   * @param config_path YAML 配置文件路径，包含 color / color_thresh 等参数
   */
  explicit Rm_Buff_Detector(const std::string & config_path);
  ~Rm_Buff_Detector();

  // 禁止拷贝，允许移动
  Rm_Buff_Detector(const Rm_Buff_Detector &) = delete;
  Rm_Buff_Detector & operator=(const Rm_Buff_Detector &) = delete;

  /**
   * @brief 对单帧图像进行神符检测
   * @param img BGR 图像 (可能被绘制调试信息)
   * @return 检测到的 PowerRune，失败时返回 std::nullopt
   */
  std::optional<PowerRune> detect(cv::Mat & img);

  std::optional<PowerRune> detect(
    cv::Mat & img, std::chrono::steady_clock::time_point timestamp);

  /**
   * @brief 使用电控上报的敌方颜色更新检测颜色
   * @param enemy_color 0 = 蓝色，1 = 红色
   *
   * 仅当配置中的 enemy_color 为 "auto" 时生效。
   */
  void set_enemy_color(uint8_t enemy_color);

  /**
   * @brief 开启/关闭调试绘制 (在 img 上绘制识别框和关键点)
   * @param enable true 时会在 detect() 调用中直接在输入图像上绘图
   */
  void set_debug_draw(bool enable);

  /**
   * @brief 查询调试绘制是否开启
   */
  [[nodiscard]] bool is_debug_draw() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__RM_BUFF_DETECTOR_HPP
