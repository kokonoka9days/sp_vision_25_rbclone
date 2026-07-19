#ifndef AUTO_DRONE__DRONE_YOLO_HPP
#define AUTO_DRONE__DRONE_YOLO_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "drone_armor.hpp"

namespace auto_drone
{

struct YOLOResult
{
  cv::Mat frame;
  std::vector<Drone> drones;
  std::chrono::steady_clock::time_point timestamp;
  std::uint64_t frame_id = 0;
  double preprocess_ms = 0.0;
  double request_ms = 0.0;
  double postprocess_ms = 0.0;
};

class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = false);
  ~YOLO();

  YOLO(const YOLO &) = delete;
  YOLO & operator=(const YOLO &) = delete;

  std::vector<Drone> detect(const cv::Mat & frame);

  // Real-time mode drops the arriving frame when both inference requests are busy.
  std::optional<YOLOResult> detect_async(
    const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
    std::uint64_t frame_id, bool drop_if_busy = true);
  std::optional<YOLOResult> flush();

  int inference_threads() const { return inference_threads_; }
  int inference_streams() const { return inference_streams_; }
  std::uint64_t dropped_frames() const { return dropped_frames_; }

private:
  struct Letterbox
  {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
  };

  struct Slot
  {
    ov::InferRequest infer_request;
    cv::Mat input;
    cv::Mat frame;
    Letterbox letterbox;
    std::chrono::steady_clock::time_point timestamp;
    std::chrono::steady_clock::time_point submitted_at;
    std::uint64_t frame_id = 0;
    double preprocess_ms = 0.0;
    bool active = false;
  };

  int input_w_ = 640;
  int input_h_ = 640;
  int num_classes_ = 1;
  int num_kpts_ = 8;
  int num_boxes_ = 0;
  int inference_threads_ = 1;
  int inference_streams_ = 2;
  float score_threshold_ = 0.70F;
  float nms_threshold_ = 0.60F;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::array<Slot, 2> slots_;
  std::uint64_t dropped_frames_ = 0;

  void prepare_slot(
    Slot & slot, const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
    std::uint64_t frame_id, bool preserve_frame);
  void start_slot(Slot & slot);
  YOLOResult finish_slot(Slot & slot);
  std::optional<std::size_t> free_slot() const;
  std::optional<std::size_t> oldest_active_slot() const;
  std::vector<Drone> postprocess(const float * output, const Letterbox & letterbox) const;
  void wait_and_discard() noexcept;
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__DRONE_YOLO_HPP
