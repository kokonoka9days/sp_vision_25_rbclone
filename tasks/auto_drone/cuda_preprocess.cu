#include "cuda_preprocess.cuh"

#include <math.h>

__global__ void preprocess_kernel(
  uint8_t * src, int src_step, int src_w, int src_h, float * dst, int dst_w, int dst_h,
  int resized_w, int resized_h, int pad_w, int pad_h)
{
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= dst_w || dy >= dst_h) return;

  const int c0 = dst_w * dst_h * 0;
  const int c1 = dst_w * dst_h * 1;
  const int c2 = dst_w * dst_h * 2;
  const int dst_idx = dy * dst_w + dx;

  if (dx >= pad_w && dx < pad_w + resized_w && dy >= pad_h && dy < pad_h + resized_h) {
    // Match the half-pixel bilinear resize used by the previous CPU preprocessing path.
    float src_x = (dx - pad_w + 0.5F) * src_w / resized_w - 0.5F;
    float src_y = (dy - pad_h + 0.5F) * src_h / resized_h - 0.5F;
    src_x = fminf(fmaxf(src_x, 0.0F), src_w - 1.0F);
    src_y = fminf(fmaxf(src_y, 0.0F), src_h - 1.0F);

    const int x0 = max(0, static_cast<int>(floorf(src_x)));
    const int y0 = max(0, static_cast<int>(floorf(src_y)));
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float u = src_x - x0;
    const float v = src_y - y0;

    const float b00 = src[y0 * src_step + x0 * 3];
    const float b10 = src[y0 * src_step + x1 * 3];
    const float b01 = src[y1 * src_step + x0 * 3];
    const float b11 = src[y1 * src_step + x1 * 3];
    const float b =
      (b00 * (1 - u) + b10 * u) * (1 - v) + (b01 * (1 - u) + b11 * u) * v;

    const float g00 = src[y0 * src_step + x0 * 3 + 1];
    const float g10 = src[y0 * src_step + x1 * 3 + 1];
    const float g01 = src[y1 * src_step + x0 * 3 + 1];
    const float g11 = src[y1 * src_step + x1 * 3 + 1];
    const float g =
      (g00 * (1 - u) + g10 * u) * (1 - v) + (g01 * (1 - u) + g11 * u) * v;

    const float r00 = src[y0 * src_step + x0 * 3 + 2];
    const float r10 = src[y0 * src_step + x1 * 3 + 2];
    const float r01 = src[y1 * src_step + x0 * 3 + 2];
    const float r11 = src[y1 * src_step + x1 * 3 + 2];
    const float r =
      (r00 * (1 - u) + r10 * u) * (1 - v) + (r01 * (1 - u) + r11 * u) * v;

    dst[c0 + dst_idx] = r / 255.0F;
    dst[c1 + dst_idx] = g / 255.0F;
    dst[c2 + dst_idx] = b / 255.0F;
  } else {
    dst[c0 + dst_idx] = 114.0F / 255.0F;
    dst[c1 + dst_idx] = 114.0F / 255.0F;
    dst[c2 + dst_idx] = 114.0F / 255.0F;
  }
}

cudaError_t launch_preprocess_kernel(
  uint8_t * src, int src_step, int src_w, int src_h, float * dst, int dst_w, int dst_h,
  int resized_w, int resized_h, int pad_w, int pad_h, cudaStream_t stream)
{
  const dim3 block(16, 16);
  const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  preprocess_kernel<<<grid, block, 0, stream>>>(
    src, src_step, src_w, src_h, dst, dst_w, dst_h, resized_w, resized_h, pad_w, pad_h);

  return cudaGetLastError();
}
