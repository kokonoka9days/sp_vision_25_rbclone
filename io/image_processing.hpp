#ifndef IO__IMAGE_PROCESSING_HPP
#define IO__IMAGE_PROCESSING_HPP

#include <opencv2/core.hpp>

namespace io::image_processing
{

/** @brief 构造带暗部保护的 8 位 Gamma 查找表 */
cv::Mat make_protected_gamma_lut(double gamma, double shadow_offset);

/** @brief 在 YCrCb 空间降噪，并仅对亮度通道应用暗部保护 Gamma */
void apply_luma_protected_gamma(
  cv::Mat & bgr_img, const cv::Mat & gamma_lut, double luma_denoise_sigma,
  double chroma_denoise_sigma);

}  // namespace io::image_processing

#endif  // IO__IMAGE_PROCESSING_HPP
