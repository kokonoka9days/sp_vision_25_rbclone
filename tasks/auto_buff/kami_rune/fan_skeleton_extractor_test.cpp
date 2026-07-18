#include "fan_skeleton_extractor.hpp"

/**
 * @file fan_skeleton_extractor_test.cpp
 * @brief 使用可控合成轮廓验证灯条骨架提取核心行为。
 *
 * 测试不依赖真实视频，分别覆盖旋转不变性、闭合轮廓起点不变性、二值图
 * 完整入口，以及单一连通轮廓中多根平行灯条的同时恢复。
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{

using auto_buff::kami_rune::ContourSkeletonResult;
using auto_buff::kami_rune::FanSkeletonExtractor;
using auto_buff::kami_rune::FanSkeletonParams;
using auto_buff::kami_rune::SkeletonLine;
using SkeletonFeaturePoints = auto_buff::kami_rune::SkeletonFeaturePoints;

// 轻量断言工具：测试失败时抛出带有具体原因的异常。
void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

// 先栅格化旋转矩形，再用 CHAIN_APPROX_NONE 获得与实际流程一致的稠密轮廓。
std::vector<cv::Point> dense_rotated_rectangle(
  const cv::Point2f & center, const cv::Size2f & size, float angle_deg)
{
  cv::Mat binary(320, 320, CV_8UC1, cv::Scalar(0));
  cv::Point2f vertices_f[4];
  cv::RotatedRect(center, size, angle_deg).points(vertices_f);
  std::vector<cv::Point> vertices;
  for (const cv::Point2f & point : vertices_f) vertices.emplace_back(cvRound(point.x), cvRound(point.y));
  cv::fillConvexPoly(binary, vertices, cv::Scalar(255));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  require(contours.size() == 1U, "synthetic rectangle must produce one contour");
  return contours.front();
}

// 一个矩形也可能产生短边配对，本测试只关心代表灯条长轴的最长骨架。
const SkeletonLine & longest_skeleton(const ContourSkeletonResult & result)
{
  require(!result.pairs.empty(), "no boundary pair was recovered");
  return std::max_element(
           result.pairs.begin(), result.pairs.end(),
           [](const auto & first, const auto & second) {
             return first.skeleton.length < second.skeleton.length;
           })
    ->skeleton;
}





// 验证旋转灯条的骨架方向、中心、长度以及轮廓起始下标不变性。
void verify_rotated_strip()
{
  FanSkeletonParams params;
  params.min_segment_samples = 6;
  params.max_contour_fill_ratio = 1.01;
  params.max_contour_aspect_ratio = 10.0;
  FanSkeletonExtractor extractor(params);

  const float angle_deg = 27.0F;
  const std::vector<cv::Point> contour =
    dense_rotated_rectangle({160.0F, 150.0F}, {150.0F, 24.0F}, angle_deg);
  const ContourSkeletonResult result = extractor.analyze_contour(contour, 7);
  const SkeletonLine & skeleton = longest_skeleton(result);

  const float angle_rad = angle_deg * static_cast<float>(CV_PI) / 180.0F;
  const cv::Point2f expected{std::cos(angle_rad), std::sin(angle_rad)};
  require(std::abs(skeleton.direction.dot(expected)) > 0.97F, "skeleton direction is incorrect");
  require(cv::norm(skeleton.center - cv::Point2f(160.0F, 150.0F)) < 3.0F, "skeleton center is incorrect");
  require(skeleton.length > 120.0F, "skeleton support is unexpectedly short");

  // 循环移动点数组只改变轮廓起点，不改变其几何形状和遍历方向。
  std::vector<cv::Point> shifted = contour;
  std::rotate(shifted.begin(), shifted.begin() + shifted.size() / 3, shifted.end());
  const SkeletonLine & shifted_skeleton = longest_skeleton(extractor.analyze_contour(shifted));
  require(
    std::abs(shifted_skeleton.direction.dot(expected)) > 0.97F,
    "result depends on the contour start index");
  require(
    cv::norm(shifted_skeleton.center - skeleton.center) < 1.5F,
    "cyclic contour handling changed the skeleton center");
}

// 验证 extract() 能从二值图完成轮廓筛选、边界提取和骨架恢复。
void verify_binary_entry_point()
{
  FanSkeletonParams params;
  params.min_segment_samples = 6;
  params.max_contour_fill_ratio = 1.01;
  params.max_contour_aspect_ratio = 10.0;
  FanSkeletonExtractor extractor(params);

  cv::Mat binary(320, 320, CV_8UC1, cv::Scalar(0));
  cv::rectangle(binary, cv::Rect(70, 140, 180, 28), cv::Scalar(255), cv::FILLED);
  const auto results = extractor.extract(binary);
  require(results.size() == 1U, "binary entry point did not retain the strip contour");
  require(!results.front().pairs.empty(), "binary entry point did not recover a skeleton");
}

// 模拟三根灯条通过底部横条连成一个色块，验证多边界的一对一配对能力。
void verify_connected_parallel_strips()
{
  FanSkeletonParams params;
  params.min_segment_samples = 6;
  params.max_contour_aspect_ratio = 4.0;
  FanSkeletonExtractor extractor(params);

  cv::Mat binary(380, 400, CV_8UC1, cv::Scalar(0));
  cv::rectangle(binary, cv::Rect(70, 270, 260, 30), cv::Scalar(255), cv::FILLED);
  for (int x : {90, 190, 290}) {
    cv::rectangle(binary, cv::Rect(x, 80, 20, 210), cv::Scalar(255), cv::FILLED);
  }

  const auto results = extractor.extract(binary);
  require(results.size() == 1U, "connected strips must form one candidate contour");

  // 横向连接条也可能形成合法骨架，因此只统计足够长的竖直骨架。
  std::vector<float> vertical_centers;
  for (const auto & pair : results.front().pairs) {
    const cv::Point2f direction = pair.skeleton.direction;
    if (std::abs(direction.y) > 0.97F && pair.skeleton.length > 150.0F) {
      vertical_centers.push_back(pair.skeleton.center.x);
    }
  }
  std::sort(vertical_centers.begin(), vertical_centers.end());
  require(vertical_centers.size() == 3U, "did not recover all three parallel strip skeletons");
  for (std::size_t i = 0; i < vertical_centers.size(); ++i) {
    const float expected_x = 100.0F + 100.0F * static_cast<float>(i);
    require(std::abs(vertical_centers[i] - expected_x) < 2.0F, "parallel skeleton center is incorrect");
  }
}

// 验证严格主干与灵敏侧枝补检可以同时恢复三叉戟结构。
void verify_trident_recovery()
{
  cv::Mat binary(380, 400, CV_8UC1, cv::Scalar(0));
  cv::line(binary, {200, 310}, {200, 70}, cv::Scalar(255), 24, cv::LINE_8);
  cv::line(binary, {194, 275}, {150, 185}, cv::Scalar(255), 7, cv::LINE_8);
  cv::line(binary, {150, 185}, {125, 85}, cv::Scalar(255), 7, cv::LINE_8);
  cv::line(binary, {206, 275}, {250, 185}, cv::Scalar(255), 7, cv::LINE_8);
  cv::line(binary, {250, 185}, {275, 85}, cv::Scalar(255), 7, cv::LINE_8);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  require(contours.size() == 1U, "synthetic trident must produce one connected contour");

  const SkeletonFeaturePoints detected = auto_buff::kami_rune::detect_and_show_skeleton(
    binary, contours, {}, "");
  require(
    detected.skeletons.size() >= 3U,
    "trident detector did not recover the central stem and both side branches");

  const auto endpoints = auto_buff::kami_rune::detect_and_show_gradient_endpoints(
    binary, contours, {}, {}, "");
  require(
    endpoints.endpoints.size() >= 3U,
    "gradient detector did not recover the trident terminal regions");

}

// 验证粗灯条平头的两个近似直角只生成一个中点，且短小轮廓毛刺不会新增端点。
void verify_one_endpoint_per_convex_region()
{
  cv::Mat binary(320, 360, CV_8UC1, cv::Scalar(0));
  cv::rectangle(binary, cv::Rect(60, 140, 240, 40), cv::Scalar(255), cv::FILLED);
  const std::vector<cv::Point> burr{{176, 140}, {180, 134}, {184, 140}};
  cv::fillConvexPoly(binary, burr, cv::Scalar(255));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  require(contours.size() == 1U, "strip with burr must have one contour");

  const auto result = auto_buff::kami_rune::detect_and_show_gradient_endpoints(
    binary, contours, {}, {}, "");
  require(result.endpoints.size() == 2U, "one convex end cap produced multiple endpoints");

  std::vector<cv::Point2f> endpoints = result.endpoints;
  std::sort(
    endpoints.begin(), endpoints.end(),
    [](const cv::Point2f & first, const cv::Point2f & second) { return first.x < second.x; });
  require(cv::norm(endpoints[0] - cv::Point2f(60.0F, 160.0F)) < 5.0F,
          "left double-corner midpoint is incorrect");
  require(cv::norm(endpoints[1] - cv::Point2f(299.0F, 160.0F)) < 5.0F,
          "right double-corner midpoint is incorrect");
}

cv::Point transform_point(
  const cv::Point2f & local, const cv::Point2f & center, float angle_rad)
{
  const float cosine = std::cos(angle_rad);
  const float sine = std::sin(angle_rad);
  return {
    cvRound(center.x + cosine * local.x - sine * local.y),
    cvRound(center.y + sine * local.x + cosine * local.y)};
}

void draw_rotated_trident(
  cv::Mat & binary, const cv::Point2f & center, float angle_rad)
{
  const auto point = [&](float x, float y) {
    return transform_point({x, y}, center, angle_rad);
  };
  cv::line(binary, point(0, 35), point(0, -65), cv::Scalar(255), 12, cv::LINE_8);
  cv::line(binary, point(-5, 15), point(-25, -20), cv::Scalar(255), 5, cv::LINE_8);
  cv::line(binary, point(-25, -20), point(-35, -55), cv::Scalar(255), 5, cv::LINE_8);
  cv::line(binary, point(5, 15), point(25, -20), cv::Scalar(255), 5, cv::LINE_8);
  cv::line(binary, point(25, -20), point(35, -55), cv::Scalar(255), 5, cv::LINE_8);
}

// 验证五目标圆周旋转场景不会被错误压缩成一个全局三叉戟，且允许外侧端部被裁切。
void verify_five_rotating_tridents_with_clipping()
{
  cv::Mat binary(500, 500, CV_8UC1, cv::Scalar(0));
  const cv::Point2f circle_center{250.0F, 250.0F};
  constexpr float radius = 195.0F;
  for (int index = 0; index < 5; ++index) {
    const float theta = 2.0F * static_cast<float>(CV_PI) * index / 5.0F;
    const cv::Point2f center =
      circle_center + radius * cv::Point2f(std::cos(theta), std::sin(theta));
    draw_rotated_trident(binary, center, theta - static_cast<float>(CV_PI) / 2.0F);
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  require(contours.size() == 5U, "five synthetic tridents must remain separate contours");

  const SkeletonFeaturePoints detected = auto_buff::kami_rune::detect_and_show_skeleton(
    binary, contours, {}, "");
  require(
    detected.skeletons.size() >= contours.size(),
    "multi-trident detector dropped one or more independent targets");

  const auto endpoints = auto_buff::kami_rune::detect_and_show_gradient_endpoints(
    binary, contours, {}, {}, "");
  require(
    endpoints.endpoints.size() >= contours.size(),
    "gradient detector dropped one or more independent clipped targets");
}

}  // namespace

namespace auto_buff::kami_rune::test
{



}  // namespace auto_buff::kami_rune::test

int main(int argc, char ** argv)
{
  // 任一子测试失败都会返回非零状态，便于 CTest 和持续集成识别。
  try {
    verify_rotated_strip();
    verify_binary_entry_point();
    verify_connected_parallel_strips();
    verify_trident_recovery();
    verify_one_endpoint_per_convex_region();
    verify_five_rotating_tridents_with_clipping();

    // 手动可视化入口：读取灰度图后重新二值化，避免压缩格式产生非 0/255 灰度值。
    if (argc == 3 && std::string(argv[1]) == "--show") {
      const cv::Mat gray = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
      if (gray.empty()) throw std::runtime_error("failed to read binary image");
      cv::Mat binary;
      cv::threshold(gray, binary, 0.0, 255.0, cv::THRESH_BINARY);
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
      const SkeletonFeaturePoints points =
        auto_buff::kami_rune::detect_and_show_skeleton(binary, contours);
      std::cout << "endpoints: " << points.endpoints.size()
                << ", corners: " << points.corners.size() << '\n';
      cv::waitKey(0);
    } else if (argc != 1) {
      throw std::invalid_argument("usage: kami_rune_fan_skeleton_test [--show binary_image]");
    }
  } catch (const std::exception & error) {
    std::cerr << "fan_skeleton_extractor_test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "fan_skeleton_extractor_test passed\n";
  return 0;
}
