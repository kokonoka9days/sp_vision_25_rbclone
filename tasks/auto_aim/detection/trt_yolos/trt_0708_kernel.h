#ifndef PREPROCESS_KERNEL_H
#define PREPROCESS_KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 在 CUDA 流上启动图像预处理 @param src GPU 原始图像 @param src_w 原图宽度 @param src_h 原图高度 @param dst GPU 网络输入缓冲区 @param dst_w 网络输入宽度 @param dst_h 网络输入高度 @param stream CUDA 流 */
void launch_preprocess(const unsigned char* src, int src_w, int src_h,
                       float* dst, int dst_w, int dst_h,
                       cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
