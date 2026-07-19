#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ? |             | 输出命令行参数说明 }"
  "{config-path c  | ../configs/xiaohei.yaml | YAML 配置文件路径 }"
  "{start-index s  | 0           | 视频起始帧下标 }"
  "{end-index e    | 0           | 视频结束帧下标，0 表示读到结尾 }"
  "{fps f          | 60          | 固定回放帧率，用于生成 Tracker 时间戳 }"
  "{headless       | true        | 无窗口健康检查模式 }"
  "{imshow         | true       | 显示回放窗口，覆盖 headless 设置 }"
  "{require-tracking | true      | 是否要求至少进入一次 tracking }"
  "{@video-path    | ../快/cs.avi| AVI 视频路径 }";

namespace
{
bool finite_target(const auto_aim::Target & target)
{
  if (
    !target.center().allFinite() || !target.velocity().allFinite() ||
    !target.rotation().allFinite() || !std::isfinite(target.yaw()) ||
    !std::isfinite(target.yaw_rate()) || !target.covariance().allFinite() || target.diverged()) {
    return false;
  }
  for (int id = 0; id < target.armor_count(); ++id) {
    const double radius = target.radius(id);
    if (!std::isfinite(radius)) return false;
    if (
      target.name != auto_aim::ArmorName::base &&
      (radius < auto_aim::motion_model::MIN_RADIUS ||
       radius > auto_aim::motion_model::MAX_RADIUS)) {
      return false;
    }
  }
  for (const auto & pose : target.armor_pose_list()) {
    if (!pose.matrix().allFinite()) return false;
  }
  return true;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }

  const std::string video_path = cli.get<std::string>(0);
  const std::string config_path = cli.get<std::string>("config-path");
  const int start_index = cli.get<int>("start-index");
  const int end_index = cli.get<int>("end-index");
  const double fps = cli.get<double>("fps");
  const bool headless = cli.get<bool>("headless");
  const bool show_image = cli.get<bool>("imshow") || !headless;
  const bool require_tracking = cli.get<bool>("require-tracking");
  if (start_index < 0 || end_index < 0 || fps <= 0 || !std::isfinite(fps)) {
    tools::logger()->error("start-index/end-index must be non-negative and fps must be positive");
    return 2;
  }

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("Cannot open AVI video: {}", video_path);
    return 2;
  }

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, &solver);
  solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  const auto timestamp_origin = std::chrono::steady_clock::now();
  int processed_frames = 0;
  int detection_frames = 0;
  int observation_frames = 0;
  bool tracking_seen = false;
  bool health_ok = true;

  for (int frame_index = start_index; ; ++frame_index) {
    if (end_index > 0 && frame_index > end_index) break;
    cv::Mat frame;
    if (!video.read(frame) || frame.empty()) break;

    const auto timestamp = timestamp_origin + std::chrono::duration_cast<
      std::chrono::steady_clock::duration>(std::chrono::duration<double>(frame_index / fps));
    auto armors = yolo.detect(frame, frame_index);
    if (!armors.empty()) ++detection_frames;
    auto targets = tracker.test_track(armors, timestamp);
    ++processed_frames;

    if (!targets.empty()) {
      const auto & target = targets.front();
      tracking_seen = tracking_seen || tracker.state() == "tracking";
      health_ok = health_ok && finite_target(target);
      if (target.estimator_diagnostics().observation_dim > 0) ++observation_frames;

      if (show_image) {
        for (const auto & pose : target.armor_pose_list()) {
          const auto points = solver.reproject_pose(pose, target.armor_type);
          if (points.size() == 4) {
            for (int i = 0; i < 4; ++i) {
              cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
          }
        }
      }
    }
    if (show_image) {
      cv::imshow("auto_aim_video_test", frame);
      if (cv::waitKey(100) == 'q') break;
    }
  }

  if (processed_frames == 0) health_ok = false;
  if (require_tracking && !tracking_seen) health_ok = false;
  if (show_image) cv::destroyAllWindows();
  tools::logger()->info(
    "AVI-only replay: frames={}, detection_frames={}, observation_frames={}, tracking={}",
    processed_frames, detection_frames, observation_frames, tracking_seen);
  return health_ok ? 0 : 3;
}
