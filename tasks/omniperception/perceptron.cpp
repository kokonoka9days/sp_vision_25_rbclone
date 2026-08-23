#include "perceptron.hpp"

#include <chrono>
#include <memory>
#include <thread>

#include "io/camera/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/detection/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

namespace omniperception
{
namespace
{
class USBCameraFrameSource final : public IFrameSource
{
public:
  /** @brief 将 USB 相机适配为帧源 @param camera 非拥有相机指针 */
  explicit USBCameraFrameSource(io::USBCamera * camera) : camera_(camera)
  {
    if (camera_ != nullptr) name_ = camera_->device_name;
  }

  /** @brief 在限定时间内读取一帧 @param image 输出图像 @param timestamp 输出时间戳 @param timeout 最长等待时间 @return 成功读取时返回 true */
  bool read_for(
    cv::Mat & image, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) override
  {
    return camera_ != nullptr && camera_->read_for(image, timestamp, timeout);
  }

  /** @brief 获取相机设备名 @return 设备名 */
  const std::string & name() const override { return name_; }

private:
  io::USBCamera * camera_;
  std::string name_;
};

/** @brief 将四个 USB 相机指针包装为帧源 @param camera1 相机一 @param camera2 相机二 @param camera3 相机三 @param camera4 相机四 @return 帧源列表 */
std::vector<std::shared_ptr<IFrameSource>> make_frame_sources(
  io::USBCamera * camera1, io::USBCamera * camera2, io::USBCamera * camera3,
  io::USBCamera * camera4)
{
  return {
    std::make_shared<USBCameraFrameSource>(camera1),
    std::make_shared<USBCameraFrameSource>(camera2),
    std::make_shared<USBCameraFrameSource>(camera3),
    std::make_shared<USBCameraFrameSource>(camera4)};
}

/** @brief 创建四个独立装甲板检测器 @param config_path YAML 配置路径 @return 检测器列表 */
std::vector<std::shared_ptr<auto_aim::IArmorDetector>> make_detectors(
  const std::string & config_path)
{
  std::vector<std::shared_ptr<auto_aim::IArmorDetector>> detectors;
  detectors.reserve(4);
  for (int i = 0; i < 4; ++i) {
    detectors.push_back(std::make_shared<auto_aim::YOLO>(config_path, false));
  }
  return detectors;
}
}  // namespace

Perceptron::Perceptron(
  io::USBCamera * usbcam1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
  io::USBCamera * usbcam4, const std::string & config_path)
: Perceptron(
    make_frame_sources(usbcam1, usbcam2, usbcam3, usbcam4), make_detectors(config_path),
    config_path)
{
}

Perceptron::Perceptron(
  std::vector<std::shared_ptr<IFrameSource>> frame_sources,
  std::vector<std::shared_ptr<auto_aim::IArmorDetector>> detectors,
  const std::string & config_path)
: detection_queue_(10),
  frame_sources_(std::move(frame_sources)),
  detectors_(std::move(detectors)),
  decider_(config_path),
  stop_flag_(false)
{
  if (frame_sources_.size() != detectors_.size()) {
    throw std::invalid_argument("Perceptron requires one detector per frame source");
  }
  for (std::size_t i = 0; i < frame_sources_.size(); ++i) {
    threads_.emplace_back([this, source = frame_sources_[i], detector = detectors_[i]] {
      parallel_infer_source(source, detector);
    });
  }

  tools::logger()->info("Perceptron initialized.");
}

Perceptron::~Perceptron()
{
  stop_flag_.store(true);
  detection_queue_.close();

  // 等待线程结束
  for (auto & t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  tools::logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  while (detection_queue_.try_pop(temp)) {
    result.push_back(std::move(temp));
  }

  return result;
}

// 将并行推理逻辑移动到类成员函数
void Perceptron::parallel_infer(
  io::USBCamera * cam, std::shared_ptr<auto_aim::YOLO> yolov8_parallel)
{
  parallel_infer_source(
    std::make_shared<USBCameraFrameSource>(cam), std::move(yolov8_parallel));
}

void Perceptron::parallel_infer_source(
  std::shared_ptr<IFrameSource> source,
  std::shared_ptr<auto_aim::IArmorDetector> detector)
{
  if (!source || !detector) {
    tools::logger()->error("Perceptron source or detector is null");
    return;
  }
  try {
    while (!stop_flag_.load()) {
      cv::Mat usb_img;
      std::chrono::steady_clock::time_point ts;

      if (!source->read_for(usb_img, ts, std::chrono::milliseconds(50))) continue;
      if (usb_img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        continue;
      }

      auto armors = detector->detect(usb_img);
      if (!armors.empty()) {
        auto delta_angle = decider_.delta_angle(armors, source->name());

        DetectionResult dr;
        dr.armors = std::move(armors);
        dr.timestamp = ts;
        dr.delta_yaw = delta_angle[0] / 57.3;
        dr.delta_pitch = delta_angle[1] / 57.3;
        detection_queue_.push(dr);  // 推入线程安全队列
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Exception in parallel_infer: {}", e.what());
  }
}

}  // namespace omniperception
