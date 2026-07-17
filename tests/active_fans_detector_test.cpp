#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/active_fans_detector.hpp"

namespace
{
using Detector = auto_buff::big::ActiveFanDetector;

bool near(const cv::Point2f & lhs, const cv::Point2f & rhs, float tolerance = 1.0f)
{
  return cv::norm(lhs - rhs) <= tolerance;
}

const Detector::IndexedContour * find_contour(
  const Detector::DetectionResult & result, int id)
{
  for (const auto & contour : result.detected_contours) {
    if (contour.id == id) return &contour;
  }
  return nullptr;
}

bool check_result(
  const Detector::DetectionResult & result, const cv::Point2f & expected_excluded_center)
{
  if (result.detected_contours.size() != 4 || result.outside_big_roi_ids.size() != 1 ||
      result.roi_excluded_ids.size() != 1 || result.roi_excluded_ids[0] < 0 ||
      result.remaining_contours.size() != 2) {
    return false;
  }

  const auto * excluded = find_contour(result, result.roi_excluded_ids[0]);
  const auto * outside = find_contour(result, result.outside_big_roi_ids[0]);
  return excluded != nullptr && outside != nullptr &&
         near(excluded->center, expected_excluded_center) && near(outside->center, {10.0f, 10.0f});
}
}  // namespace

int main()
{
  cv::Mat image = cv::Mat::zeros(200, 200, CV_8UC3);
  cv::circle(image, {10, 10}, 4, {255, 0, 0}, cv::FILLED);
  cv::circle(image, {100, 100}, 5, {255, 0, 0}, cv::FILLED);
  cv::circle(image, {125, 100}, 10, {255, 0, 0}, cv::FILLED);
  cv::circle(image, {150, 150}, 6, {255, 0, 0}, cv::FILLED);
  cv::circle(image, {40, 150}, 8, {0, 0, 255}, cv::FILLED);

  const std::string largest_config = "/tmp/active_fans_detector_largest.yaml";
  {
    std::ofstream config(largest_config);
    config << "active_fan_detector:\n"
              "  roi_scale: 3.0\n"
              "  buff_color: blue\n"
              "  color_thresh: 50\n"
              "  roi_contour_selection: largest_area\n";
  }

  const Detector::Roi r_roi(75.0f, 75.0f, 50.0f, 50.0f);
  const std::vector<Detector::Roi> rois{{80.0f, 80.0f, 55.0f, 40.0f}};
  Detector largest_detector(largest_config);
  const auto largest = largest_detector.detect(image, r_roi, rois);
  if (largest.binary.at<std::uint8_t>(100, 100) != 255 ||
      largest.binary.at<std::uint8_t>(150, 40) != 0 ||
      !check_result(largest, {125.0f, 100.0f})) {
    std::cerr << "largest-area contour selection failed\n";
    return 1;
  }

  const std::string nearest_config = "/tmp/active_fans_detector_nearest.yaml";
  {
    std::ofstream config(nearest_config);
    config << "active_fan_detector:\n"
              "  roi_scale: 3.0\n"
              "  buff_color: blue\n"
              "  color_thresh: 50\n"
              "  roi_contour_selection: nearest_center\n";
  }

  Detector nearest_detector(nearest_config);
  const auto nearest = nearest_detector.detect(image, r_roi, rois);
  if (!check_result(nearest, {100.0f, 100.0f})) {
    std::cerr << "nearest-center contour selection failed\n";
    return 1;
  }

  return 0;
}
