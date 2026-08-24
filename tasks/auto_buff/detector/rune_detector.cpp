#include "rune_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "json.hpp"
#include "tools/logger.hpp"

#if defined(SP_AUTO_BUFF_OPENVINO)
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#elif defined(SP_AUTO_BUFF_TENSORRT)
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include "trt_rune_detector_kernel.h"
#else
#error "RuneDetector requires SP_AUTO_BUFF_OPENVINO or SP_AUTO_BUFF_TENSORRT"
#endif

namespace auto_buff
{
namespace
{
constexpr int kClassCount = 3;
constexpr int kKeypointCount = 5;
constexpr int kKeypointDimensions = 3;
constexpr int kOutputChannels = kClassCount + kKeypointCount * kKeypointDimensions;

struct LetterboxResult
{
  cv::Mat image;
  float scale = 1.0f;
  int padding_x = 0;
  int padding_y = 0;
};

LetterboxResult centered_letterbox(const cv::Mat & source, int width, int height)
{
  LetterboxResult result;
  result.scale = std::min(
    width / static_cast<float>(source.cols), height / static_cast<float>(source.rows));
  const int resized_width = static_cast<int>(std::round(source.cols * result.scale));
  const int resized_height = static_cast<int>(std::round(source.rows * result.scale));
  result.padding_x = (width - resized_width) / 2;
  result.padding_y = (height - resized_height) / 2;

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
  cv::copyMakeBorder(
    resized, result.image, result.padding_y, height - resized_height - result.padding_y,
    result.padding_x, width - resized_width - result.padding_x, cv::BORDER_CONSTANT,
    cv::Scalar(114, 114, 114));
  return result;
}

int map_class_for_mode(int model_class, BuffMode mode)
{
  if (model_class == 0) return 0;
  if (mode == BuffMode::SMALL && model_class == 1) return 1;
  if (mode == BuffMode::BIG && model_class == 2) return 1;
  return -1;
}

#if defined(SP_AUTO_BUFF_TENSORRT)
class TensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * message) noexcept override
  {
    if (severity <= Severity::kWARNING) tools::logger()->warn("[TensorRT] {}", message);
  }
};

TensorRTLogger g_trt_logger;

void check_cuda(cudaError_t status, const char * operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

std::size_t tensor_volume(const nvinfer1::Dims & dims)
{
  std::size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("dynamic TensorRT shapes are not supported");
    volume *= static_cast<std::size_t>(dims.d[i]);
  }
  return volume;
}
#endif
}  // namespace

struct RuneDetector::Backend
{
  struct Result
  {
    const float * data = nullptr;
    int dim1 = 0;
    int dim2 = 0;
    bool output_is_nca = true;
    float scale = 1.0f;
    int padding_x = 0;
    int padding_y = 0;
  };

#if defined(SP_AUTO_BUFF_OPENVINO)
  Backend()
  {
    auto model = core.read_model(J_POWER_RUNE.onnx_path().string());
    if (model->inputs().size() != 1 || model->outputs().size() != 1) {
      throw std::runtime_error("rune model must have exactly one input and one output");
    }
    const ov::Shape input_shape = model->input().get_shape();
    if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3) {
      throw std::runtime_error("rune model input must be static NCHW [1,3,H,W]");
    }
    input_height = static_cast<int>(input_shape[2]);
    input_width = static_cast<int>(input_shape[3]);

    ov::preprocess::PrePostProcessor preprocessor(model);
    preprocessor.input().tensor()
      .set_element_type(ov::element::u8)
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);
    preprocessor.input().preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale(255.0f);
    preprocessor.input().model().set_layout("NCHW");
    model = preprocessor.build();

    std::string device = "CPU";
    try {
      device = static_cast<std::string>(J_POWER_RUNE.config_["detector"]["device"]);
    } catch (const std::exception &) {
    }
    compiled_model = core.compile_model(
      model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    input_tensor = ov::Tensor(
      ov::element::u8,
      {1, static_cast<std::size_t>(input_height), static_cast<std::size_t>(input_width), 3});
    request = compiled_model.create_infer_request();
    request.set_input_tensor(input_tensor);

    const ov::Shape output_shape = compiled_model.output().get_shape();
    if (output_shape.size() != 3 || output_shape[0] != 1 ||
        (output_shape[1] != kOutputChannels && output_shape[2] != kOutputChannels)) {
      throw std::runtime_error("rune model output must contain 18 channels");
    }
    dim1 = static_cast<int>(output_shape[1]);
    dim2 = static_cast<int>(output_shape[2]);
    output_is_nca = dim1 == kOutputChannels;
  }

  Result infer(const cv::Mat & image)
  {
    LetterboxResult letterbox = centered_letterbox(image, input_width, input_height);
    if (!letterbox.image.isContinuous()) letterbox.image = letterbox.image.clone();
    std::memcpy(
      input_tensor.data<uint8_t>(), letterbox.image.data,
      letterbox.image.total() * letterbox.image.elemSize());
    request.infer();
    output_tensor = request.get_output_tensor();
    return {
      output_tensor.data<const float>(), dim1, dim2, output_is_nca, letterbox.scale,
      letterbox.padding_x, letterbox.padding_y};
  }

  ov::Core core;
  ov::CompiledModel compiled_model;
  ov::InferRequest request;
  ov::Tensor input_tensor;
  ov::Tensor output_tensor;
#elif defined(SP_AUTO_BUFF_TENSORRT)
  Backend()
  {
    const std::string path = J_POWER_RUNE.engine_path().string();
    try {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file) {
        throw std::runtime_error(
          "cannot open TensorRT rune engine: " + path +
          "; generate it with tasks/auto_buff/scripts/build_trt_engine.sh");
      }
      const std::streamsize size = file.tellg();
      if (size <= 0) throw std::runtime_error("TensorRT rune engine is empty: " + path);
      file.seekg(0, std::ios::beg);
      std::vector<char> bytes(static_cast<std::size_t>(size));
      if (!file.read(bytes.data(), size)) throw std::runtime_error("failed to read rune engine");

      initLibNvInferPlugins(&g_trt_logger, "");
      runtime = nvinfer1::createInferRuntime(g_trt_logger);
      if (!runtime) throw std::runtime_error("failed to create TensorRT runtime");
      engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
      if (!engine) throw std::runtime_error("failed to deserialize TensorRT rune engine");
      context = engine->createExecutionContext();
      if (!context) throw std::runtime_error("failed to create TensorRT execution context");

      for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char * name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          if (!input_name.empty()) throw std::runtime_error("rune engine has multiple inputs");
          input_name = name;
        } else {
          if (!output_name.empty()) throw std::runtime_error("rune engine has multiple outputs");
          output_name = name;
        }
      }
      if (input_name.empty() || output_name.empty()) {
        throw std::runtime_error("rune engine must have one input and one output");
      }
      if (engine->getTensorDataType(input_name.c_str()) != nvinfer1::DataType::kFLOAT ||
          engine->getTensorDataType(output_name.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("rune engine must expose FP32 input and output tensors");
      }

      const nvinfer1::Dims input_dims = engine->getTensorShape(input_name.c_str());
      const nvinfer1::Dims output_dims = engine->getTensorShape(output_name.c_str());
      if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
          output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
          (output_dims.d[1] != kOutputChannels && output_dims.d[2] != kOutputChannels)) {
        throw std::runtime_error(
          "rune engine tensors must be [1,3,H,W] and [1,18,A] or [1,A,18]");
      }
      input_height = input_dims.d[2];
      input_width = input_dims.d[3];
      dim1 = output_dims.d[1];
      dim2 = output_dims.d[2];
      output_is_nca = dim1 == kOutputChannels;
      input_bytes = tensor_volume(input_dims) * sizeof(float);
      output_bytes = tensor_volume(output_dims) * sizeof(float);

      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
      check_cuda(cudaMalloc(&input_device, input_bytes), "cudaMalloc(input)");
      check_cuda(cudaMalloc(&output_device, output_bytes), "cudaMalloc(output)");
      check_cuda(cudaHostAlloc(
        reinterpret_cast<void **>(&output_host), output_bytes, cudaHostAllocDefault),
        "cudaHostAlloc(output)");
      if (!context->setTensorAddress(input_name.c_str(), input_device) ||
          !context->setTensorAddress(output_name.c_str(), output_device)) {
        throw std::runtime_error("failed to bind TensorRT rune tensors");
      }
      check_cuda(cudaMemsetAsync(input_device, 0, input_bytes, stream), "cudaMemset(warmup)");
      if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT rune warmup failed");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Backend() { release(); }

  Result infer(const cv::Mat & image)
  {
    const float scale = std::min(
      input_width / static_cast<float>(image.cols), input_height / static_cast<float>(image.rows));
    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));
    const int padding_x = (input_width - resized_width) / 2;
    const int padding_y = (input_height - resized_height) / 2;
    const std::size_t source_bytes = image.total() * image.elemSize();
    if (source_bytes > image_capacity) {
      if (image_device) check_cuda(cudaFree(image_device), "cudaFree(image)");
      check_cuda(cudaMalloc(reinterpret_cast<void **>(&image_device), source_bytes), "cudaMalloc(image)");
      image_capacity = source_bytes;
    }
    check_cuda(cudaMemcpy2DAsync(
      image_device, image.cols * image.elemSize(), image.data, image.step,
      image.cols * image.elemSize(), image.rows, cudaMemcpyHostToDevice, stream),
      "cudaMemcpy2D(image)");
    launch_rune_detector_preprocess(
      image_device, image.cols, image.rows, image.cols * image.elemSize(),
      static_cast<float *>(input_device), input_width, input_height, scale, padding_x, padding_y,
      stream);
    check_cuda(cudaGetLastError(), "launch_rune_detector_preprocess");
    if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT rune inference failed");
    check_cuda(cudaMemcpyAsync(
      output_host, output_device, output_bytes, cudaMemcpyDeviceToHost, stream),
      "cudaMemcpy(output)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inference)");
    return {output_host, dim1, dim2, output_is_nca, scale, padding_x, padding_y};
  }

  void release() noexcept
  {
    if (output_device) cudaFree(output_device);
    if (input_device) cudaFree(input_device);
    if (image_device) cudaFree(image_device);
    if (output_host) cudaFreeHost(output_host);
    if (stream) cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    output_device = nullptr;
    input_device = nullptr;
    image_device = nullptr;
    output_host = nullptr;
    stream = nullptr;
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
  }

  nvinfer1::IRuntime * runtime = nullptr;
  nvinfer1::ICudaEngine * engine = nullptr;
  nvinfer1::IExecutionContext * context = nullptr;
  cudaStream_t stream = nullptr;
  void * input_device = nullptr;
  void * output_device = nullptr;
  unsigned char * image_device = nullptr;
  float * output_host = nullptr;
  std::size_t image_capacity = 0;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  std::string input_name;
  std::string output_name;
#endif
  int input_height = 0;
  int input_width = 0;
  int dim1 = 0;
  int dim2 = 0;
  bool output_is_nca = true;
};

RuneDetector::RuneDetector()
{
  confidence_threshold_ = J_POWER_RUNE.config_["detector"]["confidence_threshold"];
  keypoint_threshold_ = J_POWER_RUNE.config_["detector"]["keypoint_confidence_threshold"];
  nms_distance_threshold_ = J_POWER_RUNE.config_["detector"]["nms_distance_threshold"];
  min_valid_keypoints_ = J_POWER_RUNE.config_["detector"]["min_valid_keypoints"];
  if (confidence_threshold_ < 0.0f || confidence_threshold_ > 1.0f ||
      keypoint_threshold_ < 0.0f || keypoint_threshold_ > 1.0f ||
      nms_distance_threshold_ <= 0.0f || min_valid_keypoints_ < 1 ||
      min_valid_keypoints_ > kKeypointCount) {
    throw std::runtime_error("invalid power_rune detector configuration");
  }
  backend_ = std::make_unique<Backend>();
}

RuneDetector::~RuneDetector() = default;

std::vector<RuneDetection> RuneDetector::detect(const cv::Mat & image, BuffMode mode)
{
  if (image.empty() || image.type() != CV_8UC3) {
    throw std::invalid_argument("rune detector expects a non-empty CV_8UC3 BGR image");
  }
  const Backend::Result result = backend_->infer(image);
  return decode_tensor(
    result.data, result.dim1, result.dim2, result.output_is_nca, result.scale,
    result.padding_x, result.padding_y, image.size(), mode, confidence_threshold_,
    keypoint_threshold_, nms_distance_threshold_, min_valid_keypoints_);
}

std::vector<RuneDetection> RuneDetector::decode_tensor(
  const float * output, int dim1, int dim2, bool output_is_nca,
  float scale, int padding_x, int padding_y, const cv::Size & image_size,
  BuffMode mode, float confidence_threshold, float keypoint_threshold,
  float nms_distance_threshold, int min_valid_keypoints)
{
  const int channels = output_is_nca ? dim1 : dim2;
  const int anchors = output_is_nca ? dim2 : dim1;
  if (!output || channels != kOutputChannels || anchors <= 0 || !(scale > 0.0f)) {
    throw std::runtime_error("invalid rune detector output tensor");
  }
  const auto value_at = [&](int channel, int anchor) {
    return output_is_nca ? output[channel * anchors + anchor] : output[anchor * channels + channel];
  };

  std::vector<RuneDetection> candidates;
  std::vector<cv::Point2f> centers;
  for (int anchor = 0; anchor < anchors; ++anchor) {
    int model_class = -1;
    float confidence = 0.0f;
    for (int cls = 0; cls < kClassCount; ++cls) {
      const float score = value_at(cls, anchor);
      if (std::isfinite(score) && score > confidence) {
        confidence = score;
        model_class = cls;
      }
    }
    const int class_id = map_class_for_mode(model_class, mode);
    if (class_id < 0 || confidence < confidence_threshold) continue;

    RuneDetection detection;
    detection.class_id = class_id;
    detection.confidence = confidence;
    cv::Point2f center{};
    float confidence_sum = 0.0f;
    int valid_keypoints = 0;
    bool invalid = false;
    for (int keypoint = 0; keypoint < kKeypointCount; ++keypoint) {
      const int base = kClassCount + keypoint * kKeypointDimensions;
      float x = (value_at(base, anchor) - padding_x) / scale;
      float y = (value_at(base + 1, anchor) - padding_y) / scale;
      const float kpt_confidence = value_at(base + 2, anchor);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(kpt_confidence) ||
          x < 0.0f || y < 0.0f) {
        invalid = true;
        break;
      }
      x = std::clamp(x, 0.0f, static_cast<float>(image_size.width - 1));
      y = std::clamp(y, 0.0f, static_cast<float>(image_size.height - 1));
      detection.keypoints[keypoint] = {x, y};
      detection.keypoint_confidences[keypoint] = kpt_confidence;
      if (kpt_confidence >= keypoint_threshold) {
        center += detection.keypoints[keypoint];
        confidence_sum += kpt_confidence;
        ++valid_keypoints;
      }
    }
    if (invalid || valid_keypoints < min_valid_keypoints) continue;
    center *= 1.0f / valid_keypoints;
    detection.quality = confidence * confidence_sum / valid_keypoints;
    candidates.push_back(detection);
    centers.push_back(center);
  }

  std::vector<int> order(candidates.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return candidates[lhs].quality > candidates[rhs].quality;
  });
  std::vector<RuneDetection> kept;
  const float gate_squared = nms_distance_threshold * nms_distance_threshold;
  for (int index : order) {
    bool suppressed = false;
    for (const auto & selected : kept) {
      cv::Point2f selected_center{};
      for (const auto & point : selected.keypoints) selected_center += point;
      selected_center *= 0.2f;
      const cv::Point2f delta = centers[index] - selected_center;
      if (delta.dot(delta) < gate_squared) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(candidates[index]);
  }
  return kept;
}
}  // namespace auto_buff
