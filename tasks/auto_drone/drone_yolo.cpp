#include "drone_yolo.hpp"

#include <algorithm>
#include <cmath>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/properties.hpp>
#include <stdexcept>
#include <thread>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{
namespace
{
double milliseconds(
  std::chrono::steady_clock::time_point end, std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int default_inference_threads()
{
  const auto logical_threads = std::max(1U, std::thread::hardware_concurrency());
  const auto reserved_threads = logical_threads > 2 ? logical_threads - 2 : 1U;
  return static_cast<int>(std::min(12U, reserved_threads));
}
}  // namespace

YOLO::YOLO(const std::string & config_path, bool /*debug*/)
{
  auto yaml = tools::load(config_path);
  const auto model_path = tools::read<std::string>(yaml, "model_path");

  inference_threads_ = default_inference_threads();
  if (yaml["inference_num_threads"]) {
    const int configured_threads = yaml["inference_num_threads"].as<int>();
    if (configured_threads < 0) {
      throw std::invalid_argument("inference_num_threads must be zero or positive");
    }
    if (configured_threads > 0) inference_threads_ = configured_threads;
  }

  try {
    auto model = core_.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp(model);

    auto & input_info = ppp.input(0);
    input_info.tensor()
      .set_element_type(ov::element::u8)
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);
    input_info.preprocess()
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .convert_element_type(ov::element::f32)
      .scale(255.0F);
    input_info.model().set_layout("NCHW");
    model = ppp.build();

    const ov::AnyMap config{
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::num_streams(ov::streams::Num(1)), ov::inference_num_threads(inference_threads_)};
    compiled_model_ = core_.compile_model(model, "CPU", config);

    const auto output_shape = compiled_model_.output(0).get_shape();
    const int expected_channels = 4 + num_classes_ + num_kpts_ * 3;
    if (
      output_shape.size() != 3 || output_shape[0] != 1 ||
      output_shape[1] != static_cast<std::size_t>(expected_channels)) {
      throw std::runtime_error("Unexpected YOLO output shape; expected [1,29,num_boxes]");
    }
    num_boxes_ = static_cast<int>(output_shape[2]);
    if (num_boxes_ <= 0) throw std::runtime_error("YOLO output has no candidate boxes");

    for (auto & slot : slots_) {
      slot.infer_request = compiled_model_.create_infer_request();
      slot.input.create(input_h_, input_w_, CV_8UC3);
    }

    const auto devices = compiled_model_.get_property(ov::execution_devices);
    const auto streams = compiled_model_.get_property(ov::num_streams);
    const auto optimal_requests =
      compiled_model_.get_property(ov::optimal_number_of_infer_requests);
    tools::logger()->info(
      "auto_drone::YOLO loaded: device={}, streams={}, threads={}, optimal_requests={}, "
      "output=[1,{},{}]",
      devices.empty() ? "CPU" : devices.front(), static_cast<int32_t>(streams), inference_threads_,
      optimal_requests, expected_channels, num_boxes_);
  } catch (const std::exception & e) {
    tools::logger()->error("[YOLO] OpenVINO initialization failed: {}", e.what());
    throw;
  }
}

YOLO::~YOLO() { wait_and_discard(); }

void YOLO::prepare_slot(
  Slot & slot, const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
  std::uint64_t frame_id, bool preserve_frame)
{
  if (frame.empty()) throw std::invalid_argument("Cannot run YOLO on an empty frame");
  if (frame.type() != CV_8UC3) throw std::invalid_argument("YOLO expects a CV_8UC3 frame");
  if (slot.active) throw std::logic_error("Cannot overwrite an active inference slot");

  const auto preprocess_start = std::chrono::steady_clock::now();
  slot.letterbox.scale = std::min(
    static_cast<float>(input_w_) / static_cast<float>(frame.cols),
    static_cast<float>(input_h_) / static_cast<float>(frame.rows));

  const int resized_w =
    std::clamp(static_cast<int>(std::lround(frame.cols * slot.letterbox.scale)), 1, input_w_);
  const int resized_h =
    std::clamp(static_cast<int>(std::lround(frame.rows * slot.letterbox.scale)), 1, input_h_);
  slot.letterbox.pad_x = (input_w_ - resized_w) / 2;
  slot.letterbox.pad_y = (input_h_ - resized_h) / 2;

  slot.input.setTo(cv::Scalar(114, 114, 114));
  const cv::Rect roi(slot.letterbox.pad_x, slot.letterbox.pad_y, resized_w, resized_h);
  cv::resize(frame, slot.input(roi), roi.size(), 0.0, 0.0, cv::INTER_LINEAR);

  if (preserve_frame) {
    slot.frame = frame.clone();
  } else {
    slot.frame.release();
  }
  slot.timestamp = timestamp;
  slot.frame_id = frame_id;
  slot.preprocess_ms = milliseconds(std::chrono::steady_clock::now(), preprocess_start);
}

void YOLO::start_slot(Slot & slot)
{
  if (slot.active) throw std::logic_error("Inference slot is already active");

  ov::Tensor input_tensor(
    ov::element::u8, {1, static_cast<std::size_t>(input_h_), static_cast<std::size_t>(input_w_), 3},
    slot.input.data);
  slot.infer_request.set_input_tensor(input_tensor);
  slot.submitted_at = std::chrono::steady_clock::now();
  slot.infer_request.start_async();
  slot.active = true;
}

YOLOResult YOLO::finish_slot(Slot & slot)
{
  if (!slot.active) throw std::logic_error("Inference slot is not active");

  slot.infer_request.wait();
  const auto inference_finished = std::chrono::steady_clock::now();
  auto output_tensor = slot.infer_request.get_output_tensor();
  if (output_tensor.get_element_type() != ov::element::f32) {
    throw std::runtime_error("YOLO output tensor is not FP32");
  }

  const auto output_shape = output_tensor.get_shape();
  if (
    output_shape.size() != 3 || output_shape[0] != 1 ||
    output_shape[1] != static_cast<std::size_t>(4 + num_classes_ + num_kpts_ * 3) ||
    output_shape[2] != static_cast<std::size_t>(num_boxes_)) {
    throw std::runtime_error("YOLO output shape changed after model compilation");
  }

  const auto postprocess_start = std::chrono::steady_clock::now();
  auto drones = postprocess(output_tensor.data<const float>(), slot.letterbox);
  const auto postprocess_finished = std::chrono::steady_clock::now();

  YOLOResult result;
  result.frame = std::move(slot.frame);
  result.drones = std::move(drones);
  result.timestamp = slot.timestamp;
  result.frame_id = slot.frame_id;
  result.preprocess_ms = slot.preprocess_ms;
  result.request_ms = milliseconds(inference_finished, slot.submitted_at);
  result.postprocess_ms = milliseconds(postprocess_finished, postprocess_start);

  slot.active = false;
  return result;
}

std::vector<Drone> YOLO::postprocess(const float * output, const Letterbox & letterbox) const
{
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  std::vector<int> valid_raw_indices;
  const int keypoint_offset = 4 + num_classes_;

  for (int i = 0; i < num_boxes_; ++i) {
    float max_confidence = 0.0F;
    int max_class_id = -1;
    for (int c = 0; c < num_classes_; ++c) {
      const float confidence = output[(4 + c) * num_boxes_ + i];
      if (confidence > max_confidence) {
        max_confidence = confidence;
        max_class_id = c;
      }
    }
    if (max_confidence < score_threshold_) continue;

    const float cx = output[i];
    const float cy = output[num_boxes_ + i];
    const float width = output[2 * num_boxes_ + i];
    const float height = output[3 * num_boxes_ + i];
    const float inverse_scale = 1.0F / letterbox.scale;

    const int raw_w = static_cast<int>(std::lround(width * inverse_scale));
    const int raw_h = static_cast<int>(std::lround(height * inverse_scale));
    const int raw_x =
      static_cast<int>(std::lround((cx - letterbox.pad_x) * inverse_scale - raw_w / 2.0F));
    const int raw_y =
      static_cast<int>(std::lround((cy - letterbox.pad_y) * inverse_scale - raw_h / 2.0F));

    boxes.emplace_back(raw_x, raw_y, raw_w, raw_h);
    confidences.push_back(max_confidence);
    class_ids.push_back(max_class_id);
    valid_raw_indices.push_back(i);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::vector<Drone> results;
  results.reserve(indices.size());
  for (const int index : indices) {
    const int raw_index = valid_raw_indices[index];
    const float inverse_scale = 1.0F / letterbox.scale;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> keypoint_confidences;
    keypoints.reserve(num_kpts_);
    keypoint_confidences.reserve(num_kpts_);

    for (int keypoint = 0; keypoint < num_kpts_; ++keypoint) {
      const float x = output[(keypoint_offset + keypoint * 3) * num_boxes_ + raw_index];
      const float y = output[(keypoint_offset + keypoint * 3 + 1) * num_boxes_ + raw_index];
      const float confidence =
        output[(keypoint_offset + keypoint * 3 + 2) * num_boxes_ + raw_index];
      keypoints.emplace_back(
        (x - letterbox.pad_x) * inverse_scale, (y - letterbox.pad_y) * inverse_scale);
      keypoint_confidences.push_back(confidence);
    }

    results.emplace_back(
      class_ids[index], confidences[index], boxes[index], std::move(keypoints),
      std::move(keypoint_confidences));
  }

  if (results.empty()) return results;
  const auto best = std::max_element(
    results.begin(), results.end(),
    [](const Drone & lhs, const Drone & rhs) { return lhs.confidence < rhs.confidence; });
  return {*best};
}

std::vector<Drone> YOLO::detect(const cv::Mat & frame)
{
  if (frame.empty()) return {};
  if (oldest_active_slot()) {
    throw std::logic_error("Cannot call synchronous detect while async inference is active");
  }

  auto & slot = slots_[0];
  prepare_slot(slot, frame, std::chrono::steady_clock::now(), 0, false);
  start_slot(slot);
  return finish_slot(slot).drones;
}

std::optional<YOLOResult> YOLO::detect_async(
  const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp, std::uint64_t frame_id,bool drop_if_busy)
{
  if (frame.empty()) return std::nullopt;

  auto oldest_index = oldest_active_slot();
  if (!oldest_index) {
    auto & slot = slots_[*free_slot()];
    prepare_slot(slot, frame, timestamp, frame_id, true);
    start_slot(slot);
    return std::nullopt;
  }

  auto & oldest = slots_[*oldest_index];
  if (!oldest.infer_request.wait_for(std::chrono::milliseconds{0})) {
    if (const auto available_index = free_slot()) {
      auto & available = slots_[*available_index];
      prepare_slot(available, frame, timestamp, frame_id, true);
      start_slot(available);
      return std::nullopt;
    }
    if (drop_if_busy) {
      ++dropped_frames_;
      return std::nullopt;
    }
  }

  auto result = finish_slot(oldest);
  prepare_slot(oldest, frame, timestamp, frame_id, true);
  start_slot(oldest);
  return result;
}

std::optional<YOLOResult> YOLO::flush()
{
  const auto oldest_index = oldest_active_slot();
  if (!oldest_index) return std::nullopt;

  return finish_slot(slots_[*oldest_index]);
}

std::optional<std::size_t> YOLO::free_slot() const
{
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].active) return i;
  }
  return std::nullopt;
}

std::optional<std::size_t> YOLO::oldest_active_slot() const
{
  std::optional<std::size_t> oldest;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].active) continue;
    if (!oldest || slots_[i].frame_id < slots_[*oldest].frame_id) oldest = i;
  }
  return oldest;
}

void YOLO::wait_and_discard() noexcept
{
  for (auto & slot : slots_) {
    if (!slot.active) continue;
    try {
      slot.infer_request.wait();
    } catch (const std::exception & e) {
      tools::logger()->warn("[YOLO] Failed while draining inference request: {}", e.what());
    }
    slot.active = false;
  }
}

}  // namespace auto_drone
