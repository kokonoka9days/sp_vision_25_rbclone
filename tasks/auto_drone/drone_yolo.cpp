#include "drone_yolo.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_preprocess.cuh"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{
namespace
{
class TensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * message) noexcept override
  {
    if (severity <= Severity::kWARNING) tools::logger()->warn("[TensorRT YOLO] {}", message);
  }
};

TensorRTLogger tensor_rt_logger;

void check_cuda(cudaError_t status, const char * operation)
{
  if (status == cudaSuccess) return;
  throw std::runtime_error(
          std::string(operation) + " failed: " + cudaGetErrorString(status));
}

std::size_t tensor_volume(const nvinfer1::Dims & dimensions, const char * tensor_name)
{
  std::size_t volume = 1;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    const int64_t dimension = dimensions.d[index];
    if (dimension <= 0) {
      throw std::runtime_error(
              std::string("TensorRT engine has a dynamic or invalid dimension for ") + tensor_name);
    }
    if (volume > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
      throw std::runtime_error(std::string("TensorRT tensor is too large: ") + tensor_name);
    }
    volume *= static_cast<std::size_t>(dimension);
  }
  return volume;
}
}  // namespace

YOLO::YOLO(const std::string & config_path, bool /*debug*/)
{
  auto yaml = tools::load(config_path);
  const auto model_path = tools::read<std::string>(yaml, "model_path");

  try {
    std::ifstream engine_file(model_path, std::ios::binary | std::ios::ate);
    if (!engine_file) throw std::runtime_error("Cannot open TensorRT engine: " + model_path);

    const auto engine_size = engine_file.tellg();
    if (engine_size <= 0) throw std::runtime_error("TensorRT engine is empty: " + model_path);
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(static_cast<std::size_t>(engine_size));
    if (!engine_file.read(engine_data.data(), engine_size)) {
      throw std::runtime_error("Cannot read TensorRT engine: " + model_path);
    }

    runtime_ = nvinfer1::createInferRuntime(tensor_rt_logger);
    if (runtime_ == nullptr) throw std::runtime_error("Cannot create TensorRT runtime");
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine_ == nullptr) throw std::runtime_error("Cannot deserialize TensorRT engine");

    for (int32_t index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char * name = engine_->getIOTensorName(index);
      if (name == nullptr) throw std::runtime_error("TensorRT engine contains an unnamed tensor");
      const auto mode = engine_->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        if (!input_name_.empty()) throw std::runtime_error("YOLO engine has multiple inputs");
        input_name_ = name;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        if (!output_name_.empty()) throw std::runtime_error("YOLO engine has multiple outputs");
        output_name_ = name;
      }
    }
    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error("YOLO engine must have exactly one input and one output");
    }
    if (
      engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
      engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("YOLO TensorRT input and output tensors must use FP32 I/O");
    }

    const auto input_shape = engine_->getTensorShape(input_name_.c_str());
    const auto output_shape = engine_->getTensorShape(output_name_.c_str());
    const int expected_channels = 4 + num_classes_ + num_kpts_ * 3;
    if (
      input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != 3 ||
      input_shape.d[2] <= 0 || input_shape.d[3] <= 0) {
      throw std::runtime_error("Unexpected YOLO input shape; expected static [1,3,height,width]");
    }
    if (
      output_shape.nbDims != 3 || output_shape.d[0] != 1 ||
      output_shape.d[1] != expected_channels || output_shape.d[2] <= 0) {
      throw std::runtime_error("Unexpected YOLO output shape; expected static [1,29,num_boxes]");
    }
    input_h_ = input_shape.d[2];
    input_w_ = input_shape.d[3];
    num_boxes_ = output_shape.d[2];
    input_bytes_ = tensor_volume(input_shape, input_name_.c_str()) * sizeof(float);
    output_bytes_ = tensor_volume(output_shape, output_name_.c_str()) * sizeof(float);

    for (auto & slot : slots_) {
      slot.context = engine_->createExecutionContext();
      if (slot.context == nullptr) throw std::runtime_error("Cannot create TensorRT context");
      check_cuda(cudaStreamCreate(&slot.stream), "cudaStreamCreate");
      check_cuda(cudaEventCreate(&slot.preprocess_started), "cudaEventCreate(preprocess)");
      check_cuda(cudaEventCreate(&slot.inference_started), "cudaEventCreate(inference)");
      check_cuda(cudaEventCreate(&slot.completed), "cudaEventCreate(completed)");
      check_cuda(cudaMalloc(&slot.device_input, input_bytes_), "cudaMalloc(input)");
      check_cuda(cudaMalloc(&slot.device_output, output_bytes_), "cudaMalloc(output)");
      check_cuda(
        cudaMallocHost(reinterpret_cast<void **>(&slot.host_output), output_bytes_),
        "cudaMallocHost(output)");
      if (!slot.context->setTensorAddress(input_name_.c_str(), slot.device_input)) {
        throw std::runtime_error("Cannot bind TensorRT input tensor");
      }
      if (!slot.context->setTensorAddress(output_name_.c_str(), slot.device_output)) {
        throw std::runtime_error("Cannot bind TensorRT output tensor");
      }
    }

    tools::logger()->info(
      "auto_drone::YOLO loaded TensorRT engine: input=[1,3,{},{}], output=[1,{},{}], "
      "streams={}",
      input_h_, input_w_, expected_channels, num_boxes_, inference_streams_);
  } catch (const std::exception & e) {
    release_resources();
    tools::logger()->error("[YOLO] TensorRT initialization failed: {}", e.what());
    throw;
  }
}

YOLO::~YOLO()
{
  wait_and_discard();
  release_resources();
}

void YOLO::prepare_slot(
  Slot & slot, const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
  std::uint64_t frame_id, bool preserve_frame)
{
  if (frame.empty()) throw std::invalid_argument("Cannot run YOLO on an empty frame");
  if (frame.type() != CV_8UC3) throw std::invalid_argument("YOLO expects a CV_8UC3 frame");
  if (slot.active) throw std::logic_error("Cannot overwrite an active inference slot");

  slot.letterbox.scale = std::min(
    static_cast<float>(input_w_) / static_cast<float>(frame.cols),
    static_cast<float>(input_h_) / static_cast<float>(frame.rows));

  slot.letterbox.resized_w =
    std::clamp(static_cast<int>(std::lround(frame.cols * slot.letterbox.scale)), 1, input_w_);
  slot.letterbox.resized_h =
    std::clamp(static_cast<int>(std::lround(frame.rows * slot.letterbox.scale)), 1, input_h_);
  slot.letterbox.pad_x = (input_w_ - slot.letterbox.resized_w) / 2;
  slot.letterbox.pad_y = (input_h_ - slot.letterbox.resized_h) / 2;

  if (preserve_frame) {
    slot.frame = frame.clone();
  } else {
    slot.frame.release();
  }
  slot.timestamp = timestamp;
  slot.frame_id = frame_id;

  const cv::Mat & source = preserve_frame ? slot.frame : frame;
  const std::size_t source_bytes = source.total() * source.elemSize();
  if (source_bytes > slot.device_source_capacity) {
    if (slot.device_source != nullptr) {
      check_cuda(cudaFree(slot.device_source), "cudaFree(source)");
      slot.device_source = nullptr;
      slot.device_source_capacity = 0;
    }
    check_cuda(
      cudaMalloc(reinterpret_cast<void **>(&slot.device_source), source_bytes),
      "cudaMalloc(source)");
    slot.device_source_capacity = source_bytes;
  }

  check_cuda(cudaEventRecord(slot.preprocess_started, slot.stream), "cudaEventRecord(preprocess)");
  check_cuda(
    cudaMemcpy2DAsync(
      slot.device_source, static_cast<std::size_t>(source.cols) * source.elemSize(), source.data,
      source.step, static_cast<std::size_t>(source.cols) * source.elemSize(), source.rows,
      cudaMemcpyHostToDevice, slot.stream),
    "cudaMemcpy2DAsync(source)");
  check_cuda(
    launch_preprocess_kernel(
      slot.device_source, source.cols * static_cast<int>(source.elemSize()), source.cols,
      source.rows, static_cast<float *>(slot.device_input), input_w_, input_h_,
      slot.letterbox.resized_w, slot.letterbox.resized_h, slot.letterbox.pad_x,
      slot.letterbox.pad_y, slot.stream),
    "launch_preprocess_kernel");
  check_cuda(cudaEventRecord(slot.inference_started, slot.stream), "cudaEventRecord(inference)");
}

void YOLO::start_slot(Slot & slot)
{
  if (slot.active) throw std::logic_error("Inference slot is already active");

  if (!slot.context->enqueueV3(slot.stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }
  check_cuda(
    cudaMemcpyAsync(
      slot.host_output, slot.device_output, output_bytes_, cudaMemcpyDeviceToHost, slot.stream),
    "cudaMemcpyAsync(output)");
  check_cuda(cudaEventRecord(slot.completed, slot.stream), "cudaEventRecord(completed)");
  slot.active = true;
}

YOLOResult YOLO::finish_slot(Slot & slot)
{
  if (!slot.active) throw std::logic_error("Inference slot is not active");

  check_cuda(cudaEventSynchronize(slot.completed), "cudaEventSynchronize(completed)");
  float preprocess_ms = 0.0F;
  float request_ms = 0.0F;
  check_cuda(
    cudaEventElapsedTime(&preprocess_ms, slot.preprocess_started, slot.inference_started),
    "cudaEventElapsedTime(preprocess)");
  check_cuda(
    cudaEventElapsedTime(&request_ms, slot.inference_started, slot.completed),
    "cudaEventElapsedTime(inference)");
  slot.active = false;

  const auto postprocess_start = std::chrono::steady_clock::now();
  auto drones = postprocess(slot.host_output, slot.letterbox);
  const auto postprocess_finished = std::chrono::steady_clock::now();

  YOLOResult result;
  result.frame = std::move(slot.frame);
  result.drones = std::move(drones);
  result.timestamp = slot.timestamp;
  result.frame_id = slot.frame_id;
  result.preprocess_ms = preprocess_ms;
  result.request_ms = request_ms;
  result.postprocess_ms =
    std::chrono::duration<double, std::milli>(postprocess_finished - postprocess_start).count();

  return result;
}

bool YOLO::slot_ready(const Slot & slot) const
{
  if (!slot.active) return false;
  const cudaError_t status = cudaEventQuery(slot.completed);
  if (status == cudaSuccess) return true;
  if (status == cudaErrorNotReady) return false;
  check_cuda(status, "cudaEventQuery(completed)");
  return false;
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
  const cv::Mat & frame, std::chrono::steady_clock::time_point timestamp,
  std::uint64_t frame_id, bool drop_if_busy)
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
  if (!slot_ready(oldest)) {
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
    const cudaError_t status = cudaEventSynchronize(slot.completed);
    if (status != cudaSuccess) {
      tools::logger()->warn(
        "[YOLO] Failed while draining TensorRT stream: {}", cudaGetErrorString(status));
    }
    slot.active = false;
  }
}

void YOLO::release_resources() noexcept
{
  for (auto & slot : slots_) {
    if (slot.device_source != nullptr) cudaFree(slot.device_source);
    if (slot.device_input != nullptr) cudaFree(slot.device_input);
    if (slot.device_output != nullptr) cudaFree(slot.device_output);
    if (slot.host_output != nullptr) cudaFreeHost(slot.host_output);
    if (slot.preprocess_started != nullptr) cudaEventDestroy(slot.preprocess_started);
    if (slot.inference_started != nullptr) cudaEventDestroy(slot.inference_started);
    if (slot.completed != nullptr) cudaEventDestroy(slot.completed);
    if (slot.stream != nullptr) cudaStreamDestroy(slot.stream);
    delete slot.context;
    slot = Slot{};
  }
  delete engine_;
  engine_ = nullptr;
  delete runtime_;
  runtime_ = nullptr;
}

}  // namespace auto_drone
