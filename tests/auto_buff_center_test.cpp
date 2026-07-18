#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/active_fans_detector.hpp"
#include "tasks/auto_buff/yolo11_buff.hpp"
#include "tasks/auto_buff/kami_rune/fan_skeleton_extractor.hpp"
#include "tools/exiter.hpp"

// 可视化颜色：
// 青色：big ROI
// 红色：big ROI 外剔除轮廓
// 橙色：面积小于 rune_center 轮廓的剔除轮廓
// 紫色：深度 ROI 对应剔除轮廓
// 绿色：最终保留轮廓
// 灰色：初始编号轮廓

const std::string keys =
  "{help h usage ? | | 输出帮助}"
  "{config-path c  | ../configs/buff_test.yaml | 配置文件路径}"
  "{@input-path    | /home/rm/Desktop/gitVV/buff_avi/hik_26red_dglg | 视频文件路径}";



int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string input_path = cli.get<std::string>(0);
  const std::string config_path = cli.get<std::string>("config-path");
  cv::VideoCapture video(input_path + ".avi");
  if (!video.isOpened()) {
    std::cerr << "无法打开视频: " << input_path << ".avi\n";
    return 1;
  }

  auto_buff::YOLO11_BUFF yolo_detector(config_path);
  auto_buff::big::ActiveFanDetector active_fan_detector(config_path);

  tools::Exiter exiter;
  cv::Mat frame;
  int frame_count = 0;
  while (!exiter.exit()) {
    video >> frame;
    if (frame.empty()) break;

    cv::Mat inference_image = frame.clone();
    const auto objects = yolo_detector.get_multicandidateboxes(inference_image);
    int r_object_index = -1;
    for (std::size_t i = 0; i < objects.size(); ++i) {
      if (objects[i].label != auto_buff::RUNE_CENTER) continue;
      if (
        r_object_index < 0 ||
        objects[i].prob > objects[static_cast<std::size_t>(r_object_index)].prob) {
        r_object_index = static_cast<int>(i);
      }
    }

    cv::Mat result_image = frame.clone();
    if (r_object_index >= 0) {
      std::vector<auto_buff::big::ActiveFanDetector::Roi> rois;
      std::vector<int> roi_object_indexes;
      rois.push_back(objects[static_cast<std::size_t>(r_object_index)].rect);
      roi_object_indexes.push_back(r_object_index);
      for (std::size_t i = 0; i < objects.size(); ++i) {
        if (i == static_cast<std::size_t>(r_object_index)) continue;
        rois.push_back(objects[i].rect);
        roi_object_indexes.push_back(static_cast<int>(i));
      }

      const auto detection = active_fan_detector.detect(
        frame, objects[static_cast<std::size_t>(r_object_index)].rect, rois);
      auto_buff::big::draw_detection_result(result_image, detection);
      auto_buff::big::draw_deep_rois(
        result_image, objects, r_object_index, roi_object_indexes, detection.roi_excluded_ids);
      auto_buff::big::print_exclusion_list(frame_count, objects, roi_object_indexes, detection);
      // 每个凸起区域只生成一个端点，并返回全部通过过滤的凸起端点。
      // 绿色表示输入轮廓，红色 E 点表示端点；该路径不再识别拐点。
      const auto endpoint_features = auto_buff::kami_rune::detect_and_show_gradient_endpoints(
        detection.binary, detection.remaining_contours, {}, {}, "ActiveFan Binary");
      std::cout << "  gradient endpoints=" << endpoint_features.endpoints.size() << '\n';
    } else {
      auto_buff::big::draw_deep_rois(result_image, objects, -1, {}, {});
      std::cout << "[frame " << frame_count << "] no rune_center ROI\n";
    }

    cv::imshow("ActiveFan ROI Exclusion", result_image);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') break;
    frame_count++;
  }

  cv::destroyAllWindows();
  return 0;
}
