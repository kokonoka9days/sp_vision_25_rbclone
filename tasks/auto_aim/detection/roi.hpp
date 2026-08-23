#ifndef AUTO_AIM__ROI_HPP
#define AUTO_AIM__ROI_HPP

#include <opencv2/core.hpp>

#include <stdexcept>

namespace auto_aim
{
/** @brief 将支持 -1 宽高的 ROI 配置解析为图像内有效矩形 @param configured 配置矩形 @param image_size 图像尺寸 @return 解析后的 ROI @throws std::invalid_argument 当图像为空 @throws std::out_of_range 当 ROI 超出图像 */
inline cv::Rect resolve_roi(const cv::Rect & configured, const cv::Size & image_size)
{
  if (image_size.width <= 0 || image_size.height <= 0) {
    throw std::invalid_argument("ROI requires a non-empty image");
  }
  if (configured.x < 0 || configured.y < 0 || configured.x >= image_size.width ||
      configured.y >= image_size.height) {
    throw std::out_of_range("ROI origin is outside the image");
  }

  const int width = configured.width == -1 ? image_size.width - configured.x : configured.width;
  const int height = configured.height == -1 ? image_size.height - configured.y : configured.height;
  if (width <= 0 || height <= 0 || configured.x + width > image_size.width ||
      configured.y + height > image_size.height) {
    throw std::out_of_range("ROI dimensions are outside the image");
  }
  return {configured.x, configured.y, width, height};
}
}  // namespace auto_aim

#endif  // AUTO_AIM__ROI_HPP
