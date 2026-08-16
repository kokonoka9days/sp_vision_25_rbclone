#include <chrono>
#include <cstdlib>
#include <vector>

#include <opencv2/core.hpp>

#include "tasks/auto_aim/yolo.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 2);
  const cv::Mat image(640, 640, CV_8UC3, cv::Scalar::all(0));

  {
    auto_aim::YOLO detector(argv[1], false);
    static_cast<void>(detector.detect(image, 0));

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::chrono::steady_clock::time_point> submitted;
    std::vector<std::chrono::steady_clock::time_point> completed;
    for (int frame = 0; frame < 6; ++frame) {
      const auto timestamp = start + std::chrono::milliseconds(frame);
      submitted.push_back(timestamp);
      auto result = detector.detect(
        auto_aim::YOLOFrameData(image, Eigen::Quaterniond::Identity(), timestamp), frame);
      if (!result.is_empty) completed.push_back(result.timestamp);
    }

    CHECK(completed.size() == 5);
    for (std::size_t i = 0; i < completed.size(); ++i) CHECK(completed[i] == submitted[i]);
  }

  return 0;
}
