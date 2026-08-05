#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// 暴露出给 C++ 调用的接口
cudaError_t launch_preprocess_kernel(
  uint8_t * src, int src_step, int src_w, int src_h, float * dst, int dst_w, int dst_h,
  int resized_w, int resized_h, int pad_w, int pad_h, cudaStream_t stream);

#endif
