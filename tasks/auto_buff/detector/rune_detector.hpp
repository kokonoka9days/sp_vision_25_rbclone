#ifndef AUTO_BUFF__RUNE_DETECTOR_HPP
#define AUTO_BUFF__RUNE_DETECTOR_HPP

#include <array>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "rune_system.hpp"

namespace auto_buff
{
struct RuneDetection
{
  std::array<cv::Point2f, 5> keypoints{};
  std::array<float, 5> keypoint_confidences{};
  int class_id = 0;
  float confidence = 0.0f;
  float quality = 0.0f;
};

class RuneDetector
{
public:
  RuneDetector();
  ~RuneDetector();

  RuneDetector(const RuneDetector &) = delete;
  RuneDetector & operator=(const RuneDetector &) = delete;

  std::vector<RuneDetection> detect(const cv::Mat & image, BuffMode mode);

  static std::vector<RuneDetection> decode_tensor(
    const float * output, int dim1, int dim2, bool output_is_nca,
    float scale, int padding_x, int padding_y, const cv::Size & image_size,
    BuffMode mode, float confidence_threshold, float keypoint_threshold,
    float nms_distance_threshold, int min_valid_keypoints);

private:
  struct Backend;
  std::unique_ptr<Backend> backend_;
  float confidence_threshold_ = 0.65f;
  float keypoint_threshold_ = 0.5f;
  float nms_distance_threshold_ = 30.0f;
  int min_valid_keypoints_ = 5;

};
}  // namespace auto_buff

#endif
