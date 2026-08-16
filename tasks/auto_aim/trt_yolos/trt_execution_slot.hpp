#ifndef AUTO_AIM__TRT_EXECUTION_SLOT_HPP
#define AUTO_AIM__TRT_EXECUTION_SLOT_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace auto_aim
{
/** @brief 检查 CUDA 调用结果 @param status CUDA 状态码 @param operation 操作名称 @throws std::runtime_error 当 CUDA 调用失败 */
inline void check_cuda(cudaError_t status, const char * operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

template <typename Input>
class TRTExecutionSlot
{
public:
  /** @brief 构造空 TensorRT 执行槽 */
  TRTExecutionSlot() = default;
  /** @brief 禁止复制构造 */
  TRTExecutionSlot(const TRTExecutionSlot &) = delete;
  /** @brief 禁止复制赋值 */
  TRTExecutionSlot & operator=(const TRTExecutionSlot &) = delete;

  /** @brief 移动构造执行槽并接管 GPU 资源 @param other 源执行槽 */
  TRTExecutionSlot(TRTExecutionSlot && other) noexcept { move_from(std::move(other)); }

  /** @brief 移动赋值并接管 GPU 资源 @param other 源执行槽 @return 当前执行槽 */
  TRTExecutionSlot & operator=(TRTExecutionSlot && other) noexcept
  {
    if (this != &other) {
      reset();
      move_from(std::move(other));
    }
    return *this;
  }

  /** @brief 释放执行上下文和 CUDA 缓冲区 */
  ~TRTExecutionSlot() { reset(); }

  /** @brief 创建执行上下文并分配输入输出缓冲区 @param engine TensorRT 引擎 @param input_bytes 输入缓冲字节数 @param output_bytes 输出缓冲字节数 @throws std::invalid_argument 当引擎为空 @throws std::runtime_error 当上下文或 CUDA 资源创建失败 */
  void initialize(
    nvinfer1::ICudaEngine * engine, std::size_t input_bytes, std::size_t output_bytes)
  {
    if (engine == nullptr) throw std::invalid_argument("TensorRT execution slot requires an engine");
    reset();
    context = engine->createExecutionContext();
    if (context == nullptr) throw std::runtime_error("Failed to create TensorRT execution context");
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    check_cuda(
      cudaMalloc(reinterpret_cast<void **>(&input_device), input_bytes), "cudaMalloc(input)");
    check_cuda(
      cudaMalloc(reinterpret_cast<void **>(&output_device), output_bytes), "cudaMalloc(output)");
    check_cuda(
      cudaHostAlloc(
        reinterpret_cast<void **>(&output_host), output_bytes, cudaHostAllocDefault),
      "cudaHostAlloc(output)");
  }

  /** @brief 确保图像 GPU 缓冲区至少具有指定容量 @param bytes 所需字节数 @throws std::runtime_error 当 CUDA 分配失败 */
  void ensure_image_capacity(std::size_t bytes)
  {
    if (d_img_size >= bytes) return;
    if (d_img != nullptr) {
      check_cuda(cudaFree(d_img), "cudaFree(image)");
      d_img = nullptr;
      d_img_size = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_img), bytes), "cudaMalloc(image)");
    d_img_size = bytes;
  }

  /** @brief 释放执行槽拥有的全部资源 */
  void reset() noexcept
  {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (input_device != nullptr) cudaFree(input_device);
    if (output_device != nullptr) cudaFree(output_device);
    if (output_host != nullptr) cudaFreeHost(output_host);
    if (d_img != nullptr) cudaFree(d_img);
    delete context;

    context = nullptr;
    stream = nullptr;
    input_device = nullptr;
    output_device = nullptr;
    output_host = nullptr;
    d_img = nullptr;
    d_img_size = 0;
  }

  nvinfer1::IExecutionContext * context = nullptr;
  cudaStream_t stream = nullptr;
  unsigned char * d_img = nullptr;
  std::size_t d_img_size = 0;
  Input * input_device = nullptr;
  float * output_device = nullptr;
  float * output_host = nullptr;

private:
  /** @brief 从另一执行槽接管资源 @param other 源执行槽 */
  void move_from(TRTExecutionSlot && other) noexcept
  {
    context = std::exchange(other.context, nullptr);
    stream = std::exchange(other.stream, nullptr);
    d_img = std::exchange(other.d_img, nullptr);
    d_img_size = std::exchange(other.d_img_size, 0);
    input_device = std::exchange(other.input_device, nullptr);
    output_device = std::exchange(other.output_device, nullptr);
    output_host = std::exchange(other.output_host, nullptr);
  }
};
}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_EXECUTION_SLOT_HPP
