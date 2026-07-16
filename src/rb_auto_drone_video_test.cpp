#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "tasks/auto_drone/drone_yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{config-path c  | ../configs/auto_drone.yaml | yaml配置文件路径}"
  "{mode m         | async                  | 推理模式: async 或 sync}"
  "{display        | true                   | 是否显示标注画面}"
  "{output-path o  |                        | 可选的标注视频输出路径}"
  "{start-frame s  | 0                      | 起始帧下标}"
  "{end-frame e    | 0                      | 结束帧下标，0表示直到EOF}"
  "{@input-video   |  ../yolo_buff/123/蓝.mp4                      | 输入视频文件路径}";

struct Samples
{
  std::vector<double> values;

  void add(double value) { values.push_back(value); }

  double mean() const
  {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (const double value : values) sum += value;
    return sum / values.size();
  }

  double percentile(double fraction) const
  {
    if (values.empty()) return 0.0;
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(
      std::lround(fraction * static_cast<double>(sorted.size() - 1)));
    return sorted[index];
  }

  double maximum() const
  {
    return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
  }
};

void draw_detections(cv::Mat & frame, const std::vector<auto_drone::Drone> & drones)
{
  for (const auto & drone : drones) {
    cv::rectangle(frame, drone.box, cv::Scalar(200, 255, 0), 2);
    for (const auto & point : drone.points) {
      cv::circle(frame, point, 4, cv::Scalar(0, 255, 0), -1);
    }
    cv::putText(
      frame, fmt::format("{:.2f}", drone.confidence), drone.box.tl(),
      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
  }
}

int output_fourcc(const std::string & path)
{
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".mp4") {
    return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  }
  return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
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

  const auto input_path = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto mode = cli.get<std::string>("mode");
  const bool display = cli.get<bool>("display");
  const auto output_path = cli.get<std::string>("output-path");
  const int start_frame = cli.get<int>("start-frame");
  const int end_frame = cli.get<int>("end-frame");

  if (input_path.empty()) {
    tools::logger()->error("Input video path is required");
    return 2;
  }
  if (mode != "async" && mode != "sync") {
    tools::logger()->error("Invalid mode '{}'; expected async or sync", mode);
    return 2;
  }
  if (start_frame < 0 || end_frame < 0 || (end_frame > 0 && end_frame < start_frame)) {
    tools::logger()->error("Invalid frame range: {}..{}", start_frame, end_frame);
    return 2;
  }

  cv::VideoCapture video(input_path);
  if (!video.isOpened()) {
    tools::logger()->error("Failed to open input video: {}", input_path);
    return 2;
  }

  const int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
  double source_fps = video.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(source_fps) || source_fps <= 0.0) source_fps = 30.0;
  if (start_frame > 0 && !video.set(cv::CAP_PROP_POS_FRAMES, start_frame)) {
    tools::logger()->error("Failed to seek input video to frame {}", start_frame);
    return 2;
  }

  cv::VideoWriter writer;
  if (!output_path.empty()) {
    writer.open(
      output_path, output_fourcc(output_path), source_fps, cv::Size(width, height), true);
    if (!writer.isOpened()) {
      tools::logger()->error("Failed to open output video: {}", output_path);
      return 2;
    }
  }

  auto_drone::YOLO yolo(config_path, true);
  Samples latency_samples;
  Samples preprocess_samples;
  Samples request_samples;
  Samples postprocess_samples;

  const auto benchmark_start = std::chrono::steady_clock::now();
  auto steady_start = benchmark_start;
  std::uint64_t submitted = 0;
  std::uint64_t completed = 0;
  std::uint64_t total_detections = 0;
  std::uint64_t steady_completed = 0;
  std::optional<std::uint64_t> last_completed_frame;
  bool stop_requested = false;

  auto consume = [&](auto_drone::YOLOResult result, bool allow_input) {
    if (last_completed_frame && result.frame_id <= *last_completed_frame) {
      throw std::runtime_error("Asynchronous inference returned frames out of order");
    }
    last_completed_frame = result.frame_id;
    completed++;
    total_detections += result.drones.size();

    const auto now = std::chrono::steady_clock::now();
    const double latency_ms =
      std::chrono::duration<double, std::milli>(now - result.timestamp).count();
    latency_samples.add(latency_ms);
    preprocess_samples.add(result.preprocess_ms);
    request_samples.add(result.request_ms);
    postprocess_samples.add(result.postprocess_ms);

    if (completed == 10) steady_start = now;
    if (completed > 10) steady_completed++;
    const double steady_seconds = std::chrono::duration<double>(now - steady_start).count();
    const double processing_fps =
      steady_seconds > 0.0 ? steady_completed / steady_seconds : 0.0;

    draw_detections(result.frame, result.drones);
    tools::draw_text(
      result.frame, fmt::format("Frame: {}", result.frame_id), {30, 40}, {0, 255, 0});
    tools::draw_text(
      result.frame, fmt::format("Mode: {}", mode), {30, 80}, {0, 255, 0});
    tools::draw_text(
      result.frame, fmt::format("Processing FPS: {:.1f}", processing_fps), {30, 120},
      {0, 255, 255});
    tools::draw_text(
      result.frame, fmt::format("Latency: {:.1f} ms", latency_ms), {30, 160},
      {0, 255, 255});
    tools::draw_text(
      result.frame,
      fmt::format(
        "YOLO: {:.1f}/{:.1f}/{:.1f} ms", result.preprocess_ms, result.request_ms,
        result.postprocess_ms),
      {30, 200}, {255, 255, 0});

    if (writer.isOpened()) writer.write(result.frame);
    if (display) {
      cv::Mat preview;
      const double preview_scale = std::min(1.0, 1080.0 / result.frame.rows);
      cv::resize(result.frame, preview, {}, preview_scale, preview_scale);
      cv::imshow("Auto Drone Video Test", preview);
      if (allow_input) {
        int key = cv::waitKey(1);
        if (key == ' ') key = cv::waitKey(0);
        if (key == 'q' || key == 27) stop_requested = true;
      } else {
        cv::waitKey(1);
      }
    }
  };

  try {
    cv::Mat frame;
    for (int frame_index = start_frame; !stop_requested; ++frame_index) {
      if (end_frame > 0 && frame_index > end_frame) break;
      if (!video.read(frame) || frame.empty()) break;

      const auto timestamp = std::chrono::steady_clock::now();
      submitted++;
      if (mode == "async") {
        auto result = yolo.detect_async(frame, timestamp, static_cast<std::uint64_t>(frame_index));
        if (result) consume(std::move(*result), true);
      } else {
        const auto inference_start = std::chrono::steady_clock::now();
        auto drones = yolo.detect(frame);
        const auto inference_end = std::chrono::steady_clock::now();

        auto_drone::YOLOResult result;
        result.frame = frame.clone();
        result.drones = std::move(drones);
        result.timestamp = timestamp;
        result.frame_id = static_cast<std::uint64_t>(frame_index);
        result.request_ms =
          std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        consume(std::move(result), true);
      }
    }

    if (mode == "async") {
      while (auto result = yolo.flush()) consume(std::move(*result), false);
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Video inference failed: {}", e.what());
    return 1;
  }

  const auto benchmark_end = std::chrono::steady_clock::now();
  if (submitted == 0) {
    tools::logger()->error("Input video contains no frames in the selected range");
    return 2;
  }
  const double total_seconds =
    std::chrono::duration<double>(benchmark_end - benchmark_start).count();
  const double total_fps = total_seconds > 0.0 ? completed / total_seconds : 0.0;
  const double steady_seconds =
    std::chrono::duration<double>(benchmark_end - steady_start).count();
  const double steady_fps =
    completed > 10 && steady_seconds > 0.0 ? steady_completed / steady_seconds : total_fps;

  fmt::print(
    "Video: {}x{} @ {:.2f} FPS\n"
    "Mode: {}, inference threads: {}\n"
    "Submitted: {}, completed: {}, detections: {}\n"
    "Throughput: {:.2f} FPS total, {:.2f} FPS after warmup\n"
    "End-to-end latency: mean {:.2f} ms, P50 {:.2f} ms, P95 {:.2f} ms, max {:.2f} ms\n"
    "Stages mean: preprocess {:.2f} ms, request {:.2f} ms, postprocess {:.2f} ms\n",
    width, height, source_fps, mode, yolo.inference_threads(), submitted, completed,
    total_detections, total_fps, steady_fps, latency_samples.mean(), latency_samples.percentile(0.50),
    latency_samples.percentile(0.95), latency_samples.maximum(), preprocess_samples.mean(),
    request_samples.mean(), postprocess_samples.mean());

  if (submitted != completed) {
    tools::logger()->error(
      "Submitted/completed mismatch: {}/{}", submitted, completed);
    return 1;
  }
  return 0;
}
