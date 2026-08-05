#include "drone_armor.hpp"

#include <algorithm>
#include <numeric>

namespace auto_drone
{

Drone::Drone(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> drone_keypoints,
  std::vector<float> keypoint_confidences)
: points(std::move(drone_keypoints)),
  point_confidences(std::move(keypoint_confidences)),
  class_id(class_id),
  box(box),
  confidence(confidence)
{
  cv::Point2f sum(0, 0);
  for (const auto & pt : points) {
    sum += pt;
  }
  center = cv::Point2f(sum.x / points.size(), sum.y / points.size());

  this->color = (class_id == 0) ? Color::blue : Color::red;
  this->name = DroneName::drone;
}

Drone::Drone(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> drone_keypoints,
  cv::Point2f offset, std::vector<float> keypoint_confidences)
: points(std::move(drone_keypoints)),
  point_confidences(std::move(keypoint_confidences)),
  class_id(class_id),
  box(box),
  confidence(confidence)
{
  std::transform(
    points.begin(), points.end(), points.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });

  cv::Point2f sum(0, 0);
  for (const auto & pt : points) {
    sum += pt;
  }
  center = cv::Point2f(sum.x / points.size(), sum.y / points.size());

  this->color = (class_id == 0) ? Color::blue : Color::red;
  this->name = DroneName::drone;
}

}  // namespace auto_drone
