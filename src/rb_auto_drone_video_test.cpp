#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
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
  "{mode m         | async                  | 推理模式: async、latest 或 sync}"
  "{display        | true                   | 是否显示标注画面}"
  "{output-path o  |                        | 可选的标注视频输出路径}"
  "{start-frame s  | 1                     | 从输入视频第几帧开始识别（从1开始）}"
  "{end-frame e    | 0                      | 识别到输入视频第几帧，0表示直到EOF}"
  "{kalman-trajectory | true                 | 是否绘制卡尔曼滤波及预测轨迹}"
  "{prediction-frames | 15                   | 卡尔曼轨迹向前预测的帧数}"
  "{@input-video   |  ../快/6.avi                      | 输入视频文件路径}";

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

class ImageKalmanTrajectory
{
public:
  explicit ImageKalmanTrajectory(int prediction_frames, std::size_t history_size = 60)
  : filter_(4, 2, 0, CV_32F),
    prediction_frames_(prediction_frames),
    history_size_(history_size)
  {
    filter_.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    filter_.measurementMatrix.at<float>(0, 0) = 1.0F;
    filter_.measurementMatrix.at<float>(1, 1) = 1.0F;
    cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(25.0));
    cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(1.0));
  }

  void update(const std::vector<auto_drone::Drone> & drones, std::uint64_t frame_number)
  {
    const auto observation = select_observation(drones);
    if (!initialized_) {
      if (observation) initialize(*observation, frame_number);
      return;
    }

    const auto frame_delta = static_cast<float>(std::max<std::uint64_t>(
      1, frame_number > last_frame_number_ ? frame_number - last_frame_number_ : 1));
    configure_prediction(frame_delta);
    filter_.predict();
    last_frame_number_ = frame_number;

    if (observation) {
      cv::Mat observed(2, 1, CV_32F);
      observed.at<float>(0) = observation->center.x;
      observed.at<float>(1) = observation->center.y;
      filter_.correct(observed);
      constexpr float box_smoothing = 0.2F;
      box_size_ = box_size_ * (1.0F - box_smoothing) + observation->box_size * box_smoothing;
      missing_frames_ = 0;
    } else {
      missing_frames_ += static_cast<int>(std::lround(frame_delta));
      if (missing_frames_ > max_missing_frames_) {
        reset();
        return;
      }
    }

    append_current_position();
  }

  void draw(cv::Mat & frame) const
  {
    if (!initialized_ || history_.empty()) return;

    const cv::Scalar history_color(0, 215, 255);
    const cv::Scalar prediction_color(255, 0, 255);
    for (std::size_t i = 1; i < history_.size(); ++i) {
      draw_clipped_line(frame, history_[i - 1], history_[i], history_color, 2);
    }

    const auto current = history_.back();
    cv::circle(frame, to_pixel(current), 5, history_color, -1, cv::LINE_AA);
    cv::putText(
      frame, "KF", to_pixel(current + cv::Point2f(8.0F, -8.0F)), cv::FONT_HERSHEY_SIMPLEX,
      0.55, history_color, 2, cv::LINE_AA);

    if (prediction_frames_ <= 0) return;
    const cv::Point2f velocity(
      filter_.statePost.at<float>(2), filter_.statePost.at<float>(3));
    const cv::Point2f predicted = current + velocity * static_cast<float>(prediction_frames_);
    draw_dashed_line(frame, current, predicted, prediction_color, 2);
    draw_predicted_box(frame, predicted, box_size_, prediction_color);
    cv::drawMarker(
      frame, to_pixel(predicted), prediction_color, cv::MARKER_DIAMOND, 12, 2, cv::LINE_AA);
    cv::putText(
      frame, fmt::format("+{}f", prediction_frames_),
      to_pixel(predicted + cv::Point2f(8.0F, -8.0F)), cv::FONT_HERSHEY_SIMPLEX, 0.55,
      prediction_color, 2, cv::LINE_AA);
  }

private:
  struct Observation
  {
    cv::Point2f center;
    cv::Size2f box_size;
  };

  static constexpr int max_missing_frames_ = 30;
  cv::KalmanFilter filter_;
  int prediction_frames_;
  std::size_t history_size_;
  bool initialized_ = false;
  int missing_frames_ = 0;
  std::uint64_t last_frame_number_ = 0;
  cv::Size2f box_size_;
  std::deque<cv::Point2f> history_;

  static std::optional<Observation> select_observation(
    const std::vector<auto_drone::Drone> & drones)
  {
    if (drones.empty()) return std::nullopt;
    const auto best = std::max_element(
      drones.begin(), drones.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.confidence < rhs.confidence;
      });
    if (!std::isfinite(best->center.x) || !std::isfinite(best->center.y)) return std::nullopt;
    if (best->box.width <= 0 || best->box.height <= 0) return std::nullopt;
    return Observation{
      best->center,
      cv::Size2f(static_cast<float>(best->box.width), static_cast<float>(best->box.height))};
  }

  void initialize(const Observation & observation, std::uint64_t frame_number)
  {
    filter_.statePost =
      (cv::Mat_<float>(4, 1) << observation.center.x, observation.center.y, 0.0F, 0.0F);
    filter_.errorCovPost = cv::Mat::zeros(4, 4, CV_32F);
    filter_.errorCovPost.at<float>(0, 0) = 25.0F;
    filter_.errorCovPost.at<float>(1, 1) = 25.0F;
    filter_.errorCovPost.at<float>(2, 2) = 100.0F;
    filter_.errorCovPost.at<float>(3, 3) = 100.0F;
    initialized_ = true;
    missing_frames_ = 0;
    last_frame_number_ = frame_number;
    box_size_ = observation.box_size;
    history_.clear();
    history_.push_back(observation.center);
  }

  void configure_prediction(float dt)
  {
    filter_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
      1.0F, 0.0F, dt, 0.0F,
      0.0F, 1.0F, 0.0F, dt,
      0.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 0.0F, 1.0F);

    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;
    constexpr float acceleration_noise = 4.0F;
    filter_.processNoiseCov = acceleration_noise * (cv::Mat_<float>(4, 4) <<
      0.25F * dt4, 0.0F, 0.5F * dt3, 0.0F,
      0.0F, 0.25F * dt4, 0.0F, 0.5F * dt3,
      0.5F * dt3, 0.0F, dt2, 0.0F,
      0.0F, 0.5F * dt3, 0.0F, dt2);
  }

  void append_current_position()
  {
    history_.emplace_back(filter_.statePost.at<float>(0), filter_.statePost.at<float>(1));
    while (history_.size() > history_size_) history_.pop_front();
  }

  void reset()
  {
    initialized_ = false;
    missing_frames_ = 0;
    box_size_ = {};
    history_.clear();
  }

  static cv::Point to_pixel(const cv::Point2f & point)
  {
    constexpr float coordinate_limit = 1000000.0F;
    return {
      cvRound(std::clamp(point.x, -coordinate_limit, coordinate_limit)),
      cvRound(std::clamp(point.y, -coordinate_limit, coordinate_limit))};
  }

  static void draw_clipped_line(
    cv::Mat & frame, const cv::Point2f & start, const cv::Point2f & end,
    const cv::Scalar & color, int thickness)
  {
    auto clipped_start = to_pixel(start);
    auto clipped_end = to_pixel(end);
    if (cv::clipLine(frame.size(), clipped_start, clipped_end)) {
      cv::line(frame, clipped_start, clipped_end, color, thickness, cv::LINE_AA);
    }
  }

  static void draw_dashed_line(
    cv::Mat & frame, const cv::Point2f & start, const cv::Point2f & end,
    const cv::Scalar & color, int thickness)
  {
    const cv::Point2f delta = end - start;
    const float length = std::sqrt(delta.dot(delta));
    if (length < 1.0F) return;

    constexpr float dash_length = 10.0F;
    constexpr float gap_length = 7.0F;
    const cv::Point2f direction = delta * (1.0F / length);
    for (float offset = 0.0F; offset < length; offset += dash_length + gap_length) {
      const auto dash_start = start + direction * offset;
      const auto dash_end = start + direction * std::min(offset + dash_length, length);
      draw_clipped_line(frame, dash_start, dash_end, color, thickness);
    }
  }

  static void draw_predicted_box(
    cv::Mat & frame, const cv::Point2f & center, const cv::Size2f & size,
    const cv::Scalar & color)
  {
    if (size.width < 1.0F || size.height < 1.0F) return;
    const cv::Point2f half_size(size.width * 0.5F, size.height * 0.5F);
    const cv::Point2f top_left = center - half_size;
    const cv::Point2f top_right(center.x + half_size.x, center.y - half_size.y);
    const cv::Point2f bottom_left(center.x - half_size.x, center.y + half_size.y);
    const cv::Point2f bottom_right = center + half_size;
    draw_clipped_line(frame, top_left, top_right, color, 2);
    draw_clipped_line(frame, top_right, bottom_right, color, 2);
    draw_clipped_line(frame, bottom_right, bottom_left, color, 2);
    draw_clipped_line(frame, bottom_left, top_left, color, 2);
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
  const int start_frame_number = cli.get<int>("start-frame");
  const int end_frame_number = cli.get<int>("end-frame");
  const bool draw_kalman_trajectory = cli.get<bool>("kalman-trajectory");
  const int prediction_frames = cli.get<int>("prediction-frames");

  if (input_path.empty()) {
    tools::logger()->error("Input video path is required");
    return 2;
  }
  if (mode != "async" && mode != "latest" && mode != "sync") {
    tools::logger()->error("Invalid mode '{}'; expected async, latest or sync", mode);
    return 2;
  }
  if (
    start_frame_number < 1 || end_frame_number < 0 ||
    (end_frame_number > 0 && end_frame_number < start_frame_number)) {
    tools::logger()->error(
      "Invalid input video frame range: {}..{} (frame numbers start at 1)",
      start_frame_number, end_frame_number);
    return 2;
  }
  if (prediction_frames < 0) {
    tools::logger()->error("prediction-frames must be zero or positive");
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

  for (int skipped_frame_number = 1; skipped_frame_number < start_frame_number;
       ++skipped_frame_number) {
    if (!video.grab()) {
      tools::logger()->error(
        "Input video ended before requested start frame {}", start_frame_number);
      return 2;
    }
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
  ImageKalmanTrajectory kalman_trajectory(prediction_frames);

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
    if (draw_kalman_trajectory) {
      kalman_trajectory.update(result.drones, result.frame_id);
      kalman_trajectory.draw(result.frame);
    }
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
    for (int frame_number = start_frame_number; !stop_requested; ++frame_number) {
      if (end_frame_number > 0 && frame_number > end_frame_number) break;
      if (!video.read(frame) || frame.empty()) break;

      const auto timestamp = std::chrono::steady_clock::now();
      submitted++;
      if (mode == "async") {
        auto result = yolo.detect_async(frame, timestamp, static_cast<std::uint64_t>(frame_number));
        if (result) consume(std::move(*result), true);
      } else {
        const auto inference_start = std::chrono::steady_clock::now();
        auto drones = yolo.detect(frame);
        const auto inference_end = std::chrono::steady_clock::now();

        auto_drone::YOLOResult result;
        result.frame = frame.clone();
        result.drones = std::move(drones);
        result.timestamp = timestamp;
        result.frame_id = static_cast<std::uint64_t>(frame_number);
        result.request_ms =
          std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        consume(std::move(result), true);
      }
    }

    if (mode == "async" || mode == "latest") {
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
    "Mode: {}, TensorRT streams: {}\n"
    "Submitted: {}, completed: {}, dropped: {}, detections: {}\n"
    "Throughput: {:.2f} FPS total, {:.2f} FPS after warmup\n"
    "End-to-end latency: mean {:.2f} ms, P50 {:.2f} ms, P95 {:.2f} ms, max {:.2f} ms\n"
    "Stages mean: preprocess {:.2f} ms, request {:.2f} ms, postprocess {:.2f} ms\n",
    width, height, source_fps, mode, yolo.inference_streams(), submitted, completed,
    yolo.dropped_frames(), total_detections, total_fps, steady_fps, latency_samples.mean(),
    latency_samples.percentile(0.50), latency_samples.percentile(0.95),
    latency_samples.maximum(), preprocess_samples.mean(), request_samples.mean(),
    postprocess_samples.mean());

  if (submitted != completed + yolo.dropped_frames()) {
    tools::logger()->error(
      "Submitted/completed/dropped mismatch: {}/{}/{}", submitted, completed,
      yolo.dropped_frames());
    return 1;
  }
  return 0;
}
