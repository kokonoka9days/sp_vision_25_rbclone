#include "drone_armor.hpp" 

#include <numeric>
#include <algorithm>

namespace auto_drone
{

Drone::Drone(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> drone_keypoints)
: class_id(class_id), confidence(confidence), box(box), points(drone_keypoints)
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
  cv::Point2f offset)
: class_id(class_id), confidence(confidence), box(box), points(drone_keypoints)
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