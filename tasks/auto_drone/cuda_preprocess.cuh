#ifndef CUDA_PREPROCESS_CUH
#define CUDA_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// 暴露出给 C++ 调用的接口
void launch_preprocess_kernel(
    uint8_t* src, int src_step, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float scale, int pad_w, int pad_h,  // <--- 增加的三个仿射变换参数
    cudaStream_t stream
);

#endif