#ifndef AUTO_BUFF__RP_CORE__IMG_VIZ_HPP
#define AUTO_BUFF__RP_CORE__IMG_VIZ_HPP

#include <opencv2/core.hpp>

class ImgViz
{
public:
  static bool enabled() { return false; }
  static void enqueue_image_zero_copy(const char *, const cv::Mat &) {}
};

#endif
