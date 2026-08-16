#ifndef AUTO_BUFF_TRT_YOLO11_BUFF_KERNEL_H
#define AUTO_BUFF_TRT_YOLO11_BUFF_KERNEL_H

#include <cuda_runtime_api.h>

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 在 CUDA 流上执行 letterbox、双线性采样、BGR 转 RGB 和归一化 @param src GPU 原图 @param src_width 原图宽度 @param src_height 原图高度 @param src_step 原图行步长 @param dst GPU NCHW 输出 @param dst_width 网络宽度 @param dst_height 网络高度 @param scale 缩放系数 @param stream CUDA 流 */
void launch_yolo11_buff_preprocess(
  const unsigned char * src, int src_width, int src_height, std::size_t src_step,
  float * dst, int dst_width, int dst_height, float scale, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
