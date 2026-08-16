#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <chrono>
#include <atomic>
#include <list>
#include <memory>

#include "decider.hpp"
#include "detection.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/armor_interfaces.hpp"
#include "frame_source.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class USBCamera;
}

namespace auto_aim
{
class YOLO;
}

namespace omniperception
{

class Perceptron
{
public:
  /** @brief 使用四路 USB 相机构造全向感知器 @param usbcma1 相机一 @param usbcam2 相机二 @param usbcam3 相机三 @param usbcam4 相机四 @param config_path YAML 配置路径 */
  Perceptron(
    io::USBCamera * usbcma1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
    io::USBCamera * usbcam4, const std::string & config_path);

  /** @brief 使用抽象帧源和检测器构造全向感知器 @param frame_sources 帧源列表 @param detectors 与帧源一一对应的检测器 @param config_path YAML 配置路径 @throws std::invalid_argument 当两列表长度不一致 */
  Perceptron(
    std::vector<std::shared_ptr<IFrameSource>> frame_sources,
    std::vector<std::shared_ptr<auto_aim::IArmorDetector>> detectors,
    const std::string & config_path);

  /** @brief 停止并等待全部推理线程 */
  ~Perceptron();

  /** @brief 取出当前可用检测结果 @return 多相机检测结果列表 */
  std::vector<DetectionResult> get_detection_queue();

  /** @brief USB 相机并行推理线程入口 @param cam 非拥有相机指针 @param yolo_parallel 检测器 */
  void parallel_infer(io::USBCamera * cam, std::shared_ptr<auto_aim::YOLO> yolo_parallel);

private:
  /** @brief 抽象帧源并行推理线程入口 @param source 帧源 @param detector 检测器 */
  void parallel_infer_source(
    std::shared_ptr<IFrameSource> source,
    std::shared_ptr<auto_aim::IArmorDetector> detector);

  std::vector<std::thread> threads_;
  tools::ThreadSafeQueue<DetectionResult> detection_queue_;

  std::vector<std::shared_ptr<IFrameSource>> frame_sources_;
  std::vector<std::shared_ptr<auto_aim::IArmorDetector>> detectors_;

  Decider decider_;
  std::atomic_bool stop_flag_;
};

}  // namespace omniperception
#endif
