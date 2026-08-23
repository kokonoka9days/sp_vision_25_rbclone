#include "image_processing.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace io::image_processing
{

cv::Mat make_protected_gamma_lut(double gamma, double shadow_offset)
{
  if (!std::isfinite(gamma) || gamma <= 0.0) {
    throw std::invalid_argument("img_gamma must be finite and positive");
  }
  if (!std::isfinite(shadow_offset) || shadow_offset < 0.0 || shadow_offset > 1.0) {
    throw std::invalid_argument("img_gamma_shadow_offset must be in [0, 1]");
  }

  const double offset_value = std::pow(shadow_offset, gamma);
  const double denominator = std::pow(1.0 + shadow_offset, gamma) - offset_value;
  if (denominator <= 0.0) {
    throw std::invalid_argument("invalid protected gamma parameters");
  }

  cv::Mat lut(256, 1, CV_8U);
  for (int value = 0; value < 256; ++value) {
    const double normalized = static_cast<double>(value) / 255.0;
    const double corrected =
      (std::pow(normalized + shadow_offset, gamma) - offset_value) / denominator;
    lut.at<unsigned char>(value) = cv::saturate_cast<unsigned char>(corrected * 255.0);
  }
  return lut;
}

void apply_luma_protected_gamma(
  cv::Mat & bgr_img, const cv::Mat & gamma_lut, double luma_denoise_sigma,
  double chroma_denoise_sigma)
{
  if (bgr_img.empty()) return;
  if (bgr_img.type() != CV_8UC3) {
    throw std::invalid_argument("protected gamma expects a CV_8UC3 BGR image");
  }
  if (gamma_lut.type() != CV_8U || gamma_lut.total() != 256) {
    throw std::invalid_argument("protected gamma LUT must contain 256 unsigned bytes");
  }
  if (
    !std::isfinite(luma_denoise_sigma) || luma_denoise_sigma < 0.0 ||
    !std::isfinite(chroma_denoise_sigma) || chroma_denoise_sigma < 0.0) {
    throw std::invalid_argument("gamma denoise sigmas must be finite and non-negative");
  }

  cv::Mat ycrcb;
  cv::cvtColor(bgr_img, ycrcb, cv::COLOR_BGR2YCrCb);

  std::vector<cv::Mat> channels;
  cv::split(ycrcb, channels);
  if (luma_denoise_sigma > 0.0) {
    cv::GaussianBlur(
      channels[0], channels[0], cv::Size(3, 3), luma_denoise_sigma, luma_denoise_sigma,
      cv::BORDER_REPLICATE);
  }
  if (chroma_denoise_sigma > 0.0) {
    cv::GaussianBlur(
      channels[1], channels[1], cv::Size(3, 3), chroma_denoise_sigma,
      chroma_denoise_sigma, cv::BORDER_REPLICATE);
    cv::GaussianBlur(
      channels[2], channels[2], cv::Size(3, 3), chroma_denoise_sigma,
      chroma_denoise_sigma, cv::BORDER_REPLICATE);
  }

  cv::LUT(channels[0], gamma_lut, channels[0]);
  cv::merge(channels, ycrcb);
  cv::cvtColor(ycrcb, bgr_img, cv::COLOR_YCrCb2BGR);
}

}  // namespace io::image_processing
