#include "io/image_processing.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace
{
void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
}  // namespace

int main()
{
  const auto lut = io::image_processing::make_protected_gamma_lut(0.5, 0.04);
  require(lut.at<unsigned char>(0) == 0, "protected gamma must preserve black");
  require(lut.at<unsigned char>(255) == 255, "protected gamma must preserve white");

  for (int value = 1; value < 256; ++value) {
    require(
      lut.at<unsigned char>(value) >= lut.at<unsigned char>(value - 1),
      "protected gamma LUT must be monotonic");
  }

  const int dark_value = 4;
  const int legacy_value = cv::saturate_cast<unsigned char>(
    std::pow(static_cast<double>(dark_value) / 255.0, 0.5) * 255.0);
  require(
    lut.at<unsigned char>(dark_value) < legacy_value,
    "protected gamma must amplify deep shadows less than power gamma");

  cv::Mat noisy(9, 9, CV_8UC3, cv::Scalar::all(20));
  noisy.at<cv::Vec3b>(4, 4) = cv::Vec3b::all(60);
  cv::Mat without_denoise = noisy.clone();
  cv::Mat with_denoise = noisy.clone();
  io::image_processing::apply_luma_protected_gamma(without_denoise, lut, 0.0, 0.0);
  io::image_processing::apply_luma_protected_gamma(with_denoise, lut, 0.7, 1.0);

  require(with_denoise.type() == CV_8UC3, "processing must preserve the image type");
  require(
    with_denoise.at<cv::Vec3b>(4, 4)[0] < without_denoise.at<cv::Vec3b>(4, 4)[0],
    "pre-gamma denoising must suppress an isolated bright noise pixel");

  return 0;
}
