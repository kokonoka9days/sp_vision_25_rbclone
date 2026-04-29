#include "cuda_preprocess.cuh"
#include <algorithm>
#include <math.h>
#include <stdio.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("[CUDA Error] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)

// ==========================================
// GPU 核心计算函数
// ==========================================
__global__ void preprocess_kernel(
    uint8_t* src, int src_step, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float scale, int pad_w, int pad_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x; 
    int dy = blockIdx.y * blockDim.y + threadIdx.y; 

    if (dx >= dst_w || dy >= dst_h) return;

    int c0 = dst_w * dst_h * 0; 
    int c1 = dst_w * dst_h * 1; 
    int c2 = dst_w * dst_h * 2; 
    int dst_idx = dy * dst_w + dx; 

    int scaled_w = (int)(src_w * scale);
    int scaled_h = (int)(src_h * scale);

    if (dx >= pad_w && dx < pad_w + scaled_w && dy >= pad_h && dy < pad_h + scaled_h){

        float src_x = (dx - pad_w + 0.5f) / scale - 0.5f;
        float src_y = (dy - pad_h + 0.5f) / scale - 0.5f;

        int x0 = max(0, (int)floorf(src_x));
        int y0 = max(0, (int)floorf(src_y));
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);

        float u = src_x - x0;
        float v = src_y - y0;

        // 使用 src_step 替代简单的 w*3，彻底避免非连续内存访问越界
        float b00 = src[y0 * src_step + x0 * 3 + 0];
        float b10 = src[y0 * src_step + x1 * 3 + 0];
        float b01 = src[y1 * src_step + x0 * 3 + 0];
        float b11 = src[y1 * src_step + x1 * 3 + 0];
        float b = (b00 * (1 - u) + b10 * u) * (1 - v) + (b01 * (1 - u) + b11 * u) * v;

        float g00 = src[y0 * src_step + x0 * 3 + 1];
        float g10 = src[y0 * src_step + x1 * 3 + 1];
        float g01 = src[y1 * src_step + x0 * 3 + 1];
        float g11 = src[y1 * src_step + x1 * 3 + 1];
        float g = (g00 * (1 - u) + g10 * u) * (1 - v) + (g01 * (1 - u) + g11 * u) * v;

        float r00 = src[y0 * src_step + x0 * 3 + 2];
        float r10 = src[y0 * src_step + x1 * 3 + 2];
        float r01 = src[y1 * src_step + x0 * 3 + 2];
        float r11 = src[y1 * src_step + x1 * 3 + 2];
        float r = (r00 * (1 - u) + r10 * u) * (1 - v) + (r01 * (1 - u) + r11 * u) * v;

        dst[c0 + dst_idx] = r / 255.0f;
        dst[c1 + dst_idx] = g / 255.0f;
        dst[c2 + dst_idx] = b / 255.0f;
    } else {
        dst[c0 + dst_idx] = 114.0f / 255.0f;
        dst[c1 + dst_idx] = 114.0f / 255.0f;
        dst[c2 + dst_idx] = 114.0f / 255.0f;
    }
}

// ==========================================
// 启动函数 
// ==========================================
// 将文件底部的 launch_preprocess_kernel 完整替换为以下代码：

void launch_preprocess_kernel(
    uint8_t* src, int src_step, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float scale, int pad_w, int pad_h, // <--- 增加的三个仿射变换参数
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    // 调用核函数时，把 scale, pad_w, pad_h 一起传进去
    preprocess_kernel<<<grid, block, 0, stream>>>(
        src, src_step, src_w, src_h,
        dst, dst_w, dst_h,
        scale, pad_w, pad_h
    );
    
    CUDA_CHECK(cudaGetLastError());
}