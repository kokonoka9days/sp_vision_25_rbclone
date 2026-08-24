#include <algorithm>
#include <chrono>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/rune_debug_draw.hpp"
#include "tasks/auto_buff/rune_system.hpp"
#include "tools/logger.hpp"

namespace
{
const std::string keys =
  "{help h usage ? | | show help }"
  "{config-path c | ../configs/xiaohei.yaml | robot yaml }"
  "{mode m | big | small or big }"
  "{color | blue | red or blue target }"
  "{headless | false | disable GUI }"
  "{max-frames | 0 | stop after N frames; zero means all }"
  "{@video | /home/cyn/Desktop/sp_vision_25_rbclone/yolo_buff/123/蓝.mp4 | input video }";
}

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const std::string video_path = cli.get<std::string>(0);
  const std::string config_path = cli.get<std::string>("config-path");
  const auto mode = cli.get<std::string>("mode") == "big"
                      ? auto_buff::BuffMode::BIG
                      : auto_buff::BuffMode::SMALL;
  const auto color = cli.get<std::string>("color") == "blue"
                       ? auto_buff::EnemyColor::BLUE
                       : auto_buff::EnemyColor::RED;
  const bool headless = cli.get<bool>("headless");
  const int max_frames = cli.get<int>("max-frames");

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("[auto_buff_test] cannot open {}", video_path);
    return 1;
  }

  auto_buff::RuneSystem rune(config_path);
  const double fps = std::max(1.0, video.get(cv::CAP_PROP_FPS));
  const auto start = std::chrono::steady_clock::now();
  int frame_index = 0;
  int detection_frames = 0;
  int found_frames = 0;
  cv::Mat image;
  while (video.read(image)) {
    if (max_frames > 0 && frame_index >= max_frames) break;
    const auto timestamp = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(frame_index / fps));
    const auto command = rune.process(
      image, timestamp, Eigen::Quaterniond::Identity(), mode, color, 24.5f);
    const auto & debug = rune.debug_snapshot();
    detection_frames += debug.detections.empty() ? 0 : 1;
    found_frames += command.found ? 1 : 0;
    if (!headless) {
      auto_buff::draw_rune_debug(image, debug);
      cv::imshow("RP-26Rune replay", image);
      if (cv::waitKey(20) == 'q') break;
    }
    ++frame_index;
  }

  tools::logger()->info(
    "[auto_buff_test] frames={} detection_frames={} found_frames={}",
    frame_index, detection_frames, found_frames);
  return frame_index > 0 && detection_frames > 0 ? 0 : 2;
}
