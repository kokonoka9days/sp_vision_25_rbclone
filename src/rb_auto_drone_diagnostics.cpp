#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_drone/drone_planner.hpp"
#include "tasks/auto_drone/drone_solver.hpp"
#include "tasks/auto_drone/drone_tracker.hpp"
#include "tasks/auto_drone/drone_yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace
{

constexpr double kRadToDeg = 180.0 / CV_PI;
constexpr double kDisplayScale = 0.5;
constexpr int kPanelWidth = 600;
constexpr double kPredictionMatchToleranceS = 0.020;

double nan_value() { return std::numeric_limits<double>::quiet_NaN(); }

double rad_to_deg(double value) { return value * kRadToDeg; }

double wrapped_error_deg(double target_deg, double actual_deg)
{
  return rad_to_deg(tools::limit_rad((target_deg - actual_deg) / kRadToDeg));
}

double ray_angle_deg(const Eigen::Vector3d & lhs, const Eigen::Vector3d & rhs)
{
  const double cosine = std::clamp(lhs.normalized().dot(rhs.normalized()), -1.0, 1.0);
  return std::acos(cosine) * kRadToDeg;
}

enum class StageStatus : int
{
  unknown = -1,
  ok = 0,
  warning = 1,
  failure = 2
};

StageStatus lower_is_better(double value, double ok_limit, double warning_limit)
{
  if (!std::isfinite(value)) return StageStatus::unknown;
  if (value <= ok_limit) return StageStatus::ok;
  if (value <= warning_limit) return StageStatus::warning;
  return StageStatus::failure;
}

cv::Scalar status_color(StageStatus status)
{
  switch (status) {
    case StageStatus::ok:
      return {80, 210, 100};
    case StageStatus::warning:
      return {40, 190, 245};
    case StageStatus::failure:
      return {70, 70, 245};
    default:
      return {145, 145, 145};
  }
}

const char * status_text(StageStatus status)
{
  switch (status) {
    case StageStatus::ok:
      return "OK";
    case StageStatus::warning:
      return "WARN";
    case StageStatus::failure:
      return "FAIL";
    default:
      return "WAIT";
  }
}

struct StageRow
{
  std::string name;
  std::string metric;
  StageStatus status = StageStatus::unknown;
};

struct AngularSample
{
  std::chrono::steady_clock::time_point timestamp;
  double yaw = 0.0;
  double pitch = 0.0;
};

struct FramePacket
{
  bool valid_frame = false;
  uint64_t frame_id = 0;
  std::chrono::steady_clock::time_point timestamp{};
  std::optional<auto_drone::Target> target;
  std::optional<auto_drone::Drone> observed_drone;
  auto_drone::SolveDiagnostics solve;
  io::PoseQueryDiagnostics pose;
  std::string tracker_state = "lost";
  int image_width = 0;
  int image_height = 0;
  double capture_fps = 0.0;
  double detection_fps = 0.0;
  double detection_latency_ms = 0.0;
  double minimum_keypoint_confidence = nan_value();
  bool static_test = false;
  double world_yaw_std_deg = nan_value();
  double world_pitch_std_deg = nan_value();
  std::optional<cv::Point2f> laser_reference;
  std::optional<cv::Point2f> observed_laser_spot;
  double laser_target_error_deg = nan_value();
  double laser_spot_drift_deg = nan_value();
};

struct PendingPrediction
{
  uint64_t frame_id = 0;
  std::chrono::steady_clock::time_point target_timestamp{};
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
};

struct ControlSnapshot
{
  bool valid = false;
  uint64_t frame_id = 0;
  bool target_present = false;
  auto_drone::PlanDiagnostics plan;
  io::GimbalState gimbal_state{};
  io::DroneSendDiagnostics send;
  double command_yaw = nan_value();
  double command_pitch = nan_value();
  double gimbal_yaw_deg = nan_value();
  double gimbal_pitch_deg = nan_value();
  double yaw_tracking_error_deg = nan_value();
  double pitch_tracking_error_deg = nan_value();
  double command_stable_s = 0.0;
  double prediction_error_deg = nan_value();
  double prediction_error_m = nan_value();
};

struct MouseState
{
  std::mutex mutex;
  double scale = kDisplayScale;
  int displayed_image_width = 0;
  int displayed_image_height = 0;
  std::optional<cv::Point2f> laser_reference;
  std::optional<cv::Point2f> observed_laser_spot;
};

void mouse_callback(int event, int x, int y, int, void * userdata)
{
  auto * state = static_cast<MouseState *>(userdata);
  if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_RBUTTONDOWN) return;

  std::lock_guard<std::mutex> lock(state->mutex);
  if (
    x < 0 || y < 0 || x >= state->displayed_image_width || y >= state->displayed_image_height ||
    state->scale <= 0.0) {
    return;
  }

  const cv::Point2f raw_pixel(
    static_cast<float>(x / state->scale), static_cast<float>(y / state->scale));
  if (event == cv::EVENT_LBUTTONDOWN) {
    state->laser_reference = raw_pixel;
  } else {
    state->observed_laser_spot = raw_pixel;
  }
}

class CsvWriter
{
public:
  explicit CsvWriter(const fs::path & path)
  {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    stream_.open(path);
    if (!stream_) throw std::runtime_error("Unable to open diagnostics CSV: " + path.string());
    tools::logger()->info("[AutoDroneDiagnostics] CSV: {}", path.string());
  }

  void write(const std::map<std::string, double> & values)
  {
    if (!header_written_) {
      bool first = true;
      for (const auto & [key, value] : values) {
        static_cast<void>(value);
        if (!first) stream_ << ',';
        stream_ << key;
        first = false;
      }
      stream_ << '\n';
      header_written_ = true;
    }

    bool first = true;
    stream_ << std::setprecision(10);
    for (const auto & [key, value] : values) {
      static_cast<void>(key);
      if (!first) stream_ << ',';
      stream_ << value;
      first = false;
    }
    stream_ << '\n';

    const auto now = std::chrono::steady_clock::now();
    if (now - last_flush_ >= 1s) {
      stream_.flush();
      last_flush_ = now;
    }
  }

private:
  std::ofstream stream_;
  bool header_written_ = false;
  std::chrono::steady_clock::time_point last_flush_ = std::chrono::steady_clock::now();
};

fs::path default_csv_path()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&time, &local_time);

  std::ostringstream filename;
  filename << "auto_drone_" << std::put_time(&local_time, "%Y%m%d_%H%M%S") << ".csv";
  const fs::path base =
    fs::exists("CMakeLists.txt") ? fs::path("build/diagnostics") : fs::path("diagnostics");
  return base / filename.str();
}

void draw_cross(cv::Mat & image, const cv::Point2f & point, const cv::Scalar & color, int size = 16)
{
  cv::drawMarker(image, point, color, cv::MARKER_CROSS, size, std::max(2, size / 8), cv::LINE_AA);
}

void draw_reprojection(
  cv::Mat & image, const std::vector<cv::Point2f> & points, const cv::Scalar & color)
{
  if (points.size() != 8) return;
  const std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6},
                                               {6, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (const auto & [from, to] : edges) {
    cv::line(image, points[from], points[to], color, 2, cv::LINE_AA);
  }
}

cv::Mat draw_panel(
  int height, const std::vector<StageRow> & rows, uint64_t frame_id, bool static_test,
  const std::string & first_failure)
{
  cv::Mat panel(height, kPanelWidth, CV_8UC3, cv::Scalar(24, 27, 30));
  const cv::Scalar white(235, 238, 240);
  const cv::Scalar muted(160, 166, 172);
  tools::draw_text(panel, "AUTO DRONE PIPELINE DIAGNOSTICS", {18, 28}, white, 0.65, 2);
  tools::draw_text(
    panel, fmt::format("frame {}   static-test {}", frame_id, static_test ? "ON" : "OFF"), {18, 50},
    muted, 0.44, 1);
  tools::draw_text(
    panel, "L-click laser ref | R-click spot | s static | r reset | q quit", {18, 68}, muted, 0.34,
    1);

  const bool has_failure = !first_failure.empty();
  cv::rectangle(
    panel, cv::Rect(14, 76, kPanelWidth - 28, 32),
    has_failure ? cv::Scalar(55, 55, 180) : cv::Scalar(40, 105, 55), cv::FILLED);
  tools::draw_text(
    panel, has_failure ? "FIRST FAIL: " + first_failure : "NO VERIFIED FAILURE", {24, 98}, white,
    0.5, 1);

  const int top = 120;
  const int available = std::max(1, height - top - 8);
  const int row_height = std::max(25, available / static_cast<int>(rows.size()));
  for (size_t i = 0; i < rows.size(); ++i) {
    const int y = top + static_cast<int>(i) * row_height;
    if (y + row_height > height) break;
    const auto color = status_color(rows[i].status);
    cv::circle(panel, {22, y + row_height / 2}, 7, color, cv::FILLED, cv::LINE_AA);
    tools::draw_text(
      panel, fmt::format("{:>4}  {:<10}", status_text(rows[i].status), rows[i].name),
      {38, y + row_height / 2 + 5}, color, 0.4, 1);
    tools::draw_text(panel, rows[i].metric, {205, y + row_height / 2 + 5}, white, 0.38, 1);
    cv::line(panel, {14, y + row_height - 1}, {kPanelWidth - 14, y + row_height - 1}, {48, 52, 56});
  }
  return panel;
}

std::pair<double, double> angular_standard_deviation(const std::deque<AngularSample> & samples)
{
  if (samples.size() < 10 || samples.back().timestamp - samples.front().timestamp < 1s) {
    return {nan_value(), nan_value()};
  }

  const double reference_yaw = samples.front().yaw;
  double yaw_sum = 0.0;
  double pitch_sum = 0.0;
  for (const auto & sample : samples) {
    yaw_sum += tools::limit_rad(sample.yaw - reference_yaw);
    pitch_sum += sample.pitch;
  }
  const double yaw_mean = yaw_sum / samples.size();
  const double pitch_mean = pitch_sum / samples.size();

  double yaw_variance = 0.0;
  double pitch_variance = 0.0;
  for (const auto & sample : samples) {
    yaw_variance += std::pow(tools::limit_rad(sample.yaw - reference_yaw) - yaw_mean, 2);
    pitch_variance += std::pow(sample.pitch - pitch_mean, 2);
  }
  yaw_variance /= samples.size();
  pitch_variance /= samples.size();
  return {std::sqrt(yaw_variance) * kRadToDeg, std::sqrt(pitch_variance) * kRadToDeg};
}

void add_metric(std::map<std::string, double> & values, const std::string & key, double value)
{
  values[key] = value;
}

}  // namespace

const std::string keys =
  "{help h usage ? |                             | output command line help}"
  "{output o       |                             | diagnostics CSV path}"
  "{@config-path   | ../configs/auto_drone.yaml  | YAML configuration path}";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  fs::path csv_path = cli.get<std::string>("output");
  if (csv_path.empty()) csv_path = default_csv_path();

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_drone::YOLO yolo(config_path, true);
  auto_drone::Solver solver(config_path);
  auto_drone::Tracker tracker(config_path, &solver);
  tracker.set_gimbal(&gimbal);
  auto_drone::Planner planner(config_path);
  tools::Plotter plotter;
  CsvWriter csv(csv_path);

  tools::ThreadSafeQueue<FramePacket, true> frame_queue(1);
  frame_queue.push(FramePacket{});
  std::mutex control_snapshot_mutex;
  ControlSnapshot control_snapshot;
  std::atomic<bool> quit{false};
  std::atomic<uint64_t> reset_generation{0};

  auto control_thread = std::thread([&]() {
    const auto start_time = std::chrono::steady_clock::now();
    uint64_t last_frame_id = std::numeric_limits<uint64_t>::max();
    uint64_t consumed_reset_generation = reset_generation.load();
    std::deque<PendingPrediction> pending_predictions;
    double latest_prediction_error_deg = nan_value();
    double latest_prediction_error_m = nan_value();
    double previous_command_yaw = nan_value();
    double previous_command_pitch = nan_value();
    auto command_stable_since = std::chrono::steady_clock::now();

    while (!quit.load()) {
      const auto now = std::chrono::steady_clock::now();
      const auto packet = frame_queue.front();
      const auto gimbal_state = gimbal.state();

      if (consumed_reset_generation != reset_generation.load()) {
        consumed_reset_generation = reset_generation.load();
        pending_predictions.clear();
        latest_prediction_error_deg = nan_value();
        latest_prediction_error_m = nan_value();
      }

      auto_drone::PlanDiagnostics plan_diagnostics;
      double command_yaw = nan_value();
      double command_pitch = nan_value();
      if (packet.target.has_value()) {
        plan_diagnostics = planner.plan_diagnostics(packet.target, gimbal_state.bullet_speed);
        const auto & plan = plan_diagnostics.plan;
        command_yaw = plan.yaw * 57.3;
        command_pitch = plan.pitch * 57.3;
        gimbal.drone_send(
          plan.control, plan.fire, static_cast<float>(command_yaw), plan.yaw_vel, plan.yaw_acc,
          static_cast<float>(command_pitch), plan.pitch_vel, plan.pitch_acc);
      } else {
        // Deliberately preserve the legacy lost-target radian packet for diagnosis.
        command_yaw = gimbal_state.yaw;
        command_pitch = gimbal_state.pitch;
        gimbal.drone_send(
          false, false, static_cast<float>(command_yaw), 0.0F, 0.0F,
          static_cast<float>(command_pitch), 0.0F, 0.0F);
      }
      const auto send_diagnostics = gimbal.last_drone_send_diagnostics();
      const auto solve_value = [&](double value) {
        return packet.solve.pnp_success ? value : nan_value();
      };
      const auto plan_value = [&](double value) {
        return plan_diagnostics.plan_valid ? value : nan_value();
      };

      if (
        !std::isfinite(previous_command_yaw) ||
        std::hypot(
          wrapped_error_deg(command_yaw, previous_command_yaw),
          command_pitch - previous_command_pitch) > 0.05) {
        command_stable_since = now;
      }
      previous_command_yaw = command_yaw;
      previous_command_pitch = command_pitch;
      const double command_stable_s =
        std::chrono::duration<double>(now - command_stable_since).count();

      const double gimbal_yaw_deg = rad_to_deg(gimbal_state.yaw);
      const double gimbal_pitch_deg = rad_to_deg(gimbal_state.pitch);
      const double yaw_tracking_error =
        command_stable_s >= 0.1 ? wrapped_error_deg(command_yaw, gimbal_yaw_deg) : nan_value();
      const double pitch_tracking_error =
        command_stable_s >= 0.1 ? command_pitch - gimbal_pitch_deg : nan_value();

      const bool new_frame = packet.valid_frame && packet.frame_id != last_frame_id;
      if (new_frame) {
        if (!packet.target.has_value()) {
          pending_predictions.clear();
          latest_prediction_error_deg = nan_value();
          latest_prediction_error_m = nan_value();
        } else if (packet.solve.pnp_success) {
          while (!pending_predictions.empty() &&
                 pending_predictions.front().target_timestamp <= packet.timestamp) {
            const auto prediction = pending_predictions.front();
            pending_predictions.pop_front();
            const double time_error_s =
              std::abs(std::chrono::duration<double>(packet.timestamp - prediction.target_timestamp)
                         .count());
            if (time_error_s > kPredictionMatchToleranceS) continue;

            const auto predicted_ypd = tools::xyz2ypd(prediction.xyz);
            const auto observed_ypd = tools::xyz2ypd(packet.solve.origin_world);
            latest_prediction_error_deg = std::hypot(
              rad_to_deg(tools::limit_rad(predicted_ypd.x() - observed_ypd.x())),
              rad_to_deg(predicted_ypd.y() - observed_ypd.y()));
            latest_prediction_error_m = (prediction.xyz - packet.solve.origin_world).norm();
          }
        }

        if (
          plan_diagnostics.plan_valid &&
          plan_diagnostics.prediction_target_timestamp > packet.timestamp) {
          pending_predictions.push_back(
            {packet.frame_id, plan_diagnostics.prediction_target_timestamp,
             plan_diagnostics.predicted_xyz});
        }
        last_frame_id = packet.frame_id;
      }

      std::map<std::string, double> metrics;
      add_metric(metrics, "time_s", std::chrono::duration<double>(now - start_time).count());
      add_metric(metrics, "frame_id", static_cast<double>(packet.frame_id));
      add_metric(metrics, "capture_fps", packet.capture_fps);
      add_metric(metrics, "detection_fps", packet.detection_fps);
      add_metric(metrics, "detection_latency_ms", packet.detection_latency_ms);
      add_metric(
        metrics, "frame_age_ms",
        packet.valid_frame
          ? std::chrono::duration<double, std::milli>(now - packet.timestamp).count()
          : nan_value());
      add_metric(
        metrics, "keypoint_count",
        packet.observed_drone ? static_cast<double>(packet.observed_drone->points.size()) : 0.0);
      add_metric(metrics, "keypoint_min_confidence", packet.minimum_keypoint_confidence);
      add_metric(metrics, "aim_raw_u_px", solve_value(packet.solve.keypoint_center.x));
      add_metric(metrics, "aim_raw_v_px", solve_value(packet.solve.keypoint_center.y));
      add_metric(metrics, "aim_tvec_u_px", solve_value(packet.solve.origin_pixel.x));
      add_metric(metrics, "aim_tvec_v_px", solve_value(packet.solve.origin_pixel.y));
      add_metric(metrics, "aim_center_u_px", solve_value(packet.solve.center_pixel.x));
      add_metric(metrics, "aim_center_v_px", solve_value(packet.solve.center_pixel.y));
      add_metric(metrics, "pnp_success", packet.solve.pnp_success ? 1.0 : 0.0);
      add_metric(
        metrics, "pnp_reprojection_rmse_px", solve_value(packet.solve.reprojection_rmse_px));
      add_metric(metrics, "pnp_reprojection_max_px", solve_value(packet.solve.reprojection_max_px));
      add_metric(metrics, "pnp_depth_m", solve_value(packet.solve.origin_camera.z()));
      add_metric(metrics, "camera_x_m", solve_value(packet.solve.origin_camera.x()));
      add_metric(metrics, "camera_y_m", solve_value(packet.solve.origin_camera.y()));
      add_metric(metrics, "camera_z_m", solve_value(packet.solve.origin_camera.z()));
      add_metric(metrics, "gimbal_x_m", solve_value(packet.solve.origin_gimbal.x()));
      add_metric(metrics, "gimbal_y_m", solve_value(packet.solve.origin_gimbal.y()));
      add_metric(metrics, "gimbal_z_m", solve_value(packet.solve.origin_gimbal.z()));
      add_metric(metrics, "transform_roundtrip_px", solve_value(packet.solve.roundtrip_error_px));
      add_metric(
        metrics, "pose_sample_age_ms",
        packet.valid_frame ? packet.pose.used_sample_age_ms : nan_value());
      add_metric(
        metrics, "pose_sample_span_ms",
        packet.valid_frame ? packet.pose.sample_span_ms : nan_value());
      add_metric(
        metrics, "pose_interpolation_factor",
        packet.valid_frame ? packet.pose.interpolation_factor : nan_value());
      add_metric(
        metrics, "pose_interpolated", packet.valid_frame ? packet.pose.interpolated : nan_value());
      add_metric(
        metrics, "pose_legacy_early_return",
        packet.valid_frame ? packet.pose.legacy_early_return : nan_value());
      add_metric(metrics, "camera_offset_configured_us", camera.timestamp_offset.count());
      add_metric(metrics, "camera_offset_applied_us", 0.0);
      add_metric(metrics, "world_x_m", solve_value(packet.solve.origin_world.x()));
      add_metric(metrics, "world_y_m", solve_value(packet.solve.origin_world.y()));
      add_metric(metrics, "world_z_m", solve_value(packet.solve.origin_world.z()));
      const auto world_ypd = tools::xyz2ypd(packet.solve.origin_world);
      add_metric(metrics, "world_yaw_deg", solve_value(rad_to_deg(world_ypd.x())));
      add_metric(metrics, "world_pitch_deg", solve_value(rad_to_deg(world_ypd.y())));
      add_metric(metrics, "world_yaw_std_deg", packet.world_yaw_std_deg);
      add_metric(metrics, "world_pitch_std_deg", packet.world_pitch_std_deg);
      add_metric(
        metrics, "prediction_horizon_ms", plan_value(plan_diagnostics.prediction_horizon_s * 1e3));
      add_metric(metrics, "prediction_x_m", plan_value(plan_diagnostics.predicted_xyz.x()));
      add_metric(metrics, "prediction_y_m", plan_value(plan_diagnostics.predicted_xyz.y()));
      add_metric(metrics, "prediction_z_m", plan_value(plan_diagnostics.predicted_xyz.z()));
      add_metric(metrics, "prediction_error_deg", latest_prediction_error_deg);
      add_metric(metrics, "prediction_error_m", latest_prediction_error_m);
      add_metric(
        metrics, "angle_input_yaw_deg",
        plan_value(rad_to_deg(plan_diagnostics.input_yaw_pitch.x())));
      add_metric(
        metrics, "angle_input_pitch_deg",
        plan_value(rad_to_deg(plan_diagnostics.input_yaw_pitch.y())));
      add_metric(
        metrics, "angle_predicted_yaw_deg",
        plan_value(rad_to_deg(plan_diagnostics.predicted_yaw_pitch.x())));
      add_metric(
        metrics, "angle_predicted_pitch_deg",
        plan_value(rad_to_deg(plan_diagnostics.predicted_yaw_pitch.y())));
      add_metric(
        metrics, "angle_plan_target_yaw_deg", plan_value(plan_diagnostics.plan.target_yaw * 57.3));
      add_metric(
        metrics, "angle_plan_target_pitch_deg",
        plan_value(plan_diagnostics.plan.target_pitch * 57.3));
      add_metric(metrics, "command_yaw", command_yaw);
      add_metric(metrics, "command_pitch", command_pitch);
      add_metric(metrics, "protocol_serialized_yaw", send_diagnostics.serialized_yaw);
      add_metric(metrics, "protocol_serialized_pitch", send_diagnostics.serialized_pitch);
      add_metric(metrics, "protocol_crc_valid", send_diagnostics.crc_valid ? 1.0 : 0.0);
      add_metric(metrics, "protocol_control_encoded", send_diagnostics.control_encoded ? 1.0 : 0.0);
      add_metric(metrics, "protocol_fire_encoded", send_diagnostics.fire_encoded ? 1.0 : 0.0);
      add_metric(metrics, "gimbal_yaw_deg", gimbal_yaw_deg);
      add_metric(metrics, "gimbal_pitch_deg", gimbal_pitch_deg);
      add_metric(metrics, "control_yaw_error_deg", yaw_tracking_error);
      add_metric(metrics, "control_pitch_error_deg", pitch_tracking_error);
      add_metric(metrics, "control_command_stable_s", command_stable_s);
      add_metric(metrics, "laser_target_error_deg", packet.laser_target_error_deg);
      add_metric(metrics, "laser_spot_drift_deg", packet.laser_spot_drift_deg);

      csv.write(metrics);
      plotter.plot(nlohmann::json(metrics));

      ControlSnapshot snapshot;
      snapshot.valid = true;
      snapshot.frame_id = packet.frame_id;
      snapshot.target_present = packet.target.has_value();
      snapshot.plan = plan_diagnostics;
      snapshot.gimbal_state = gimbal_state;
      snapshot.send = send_diagnostics;
      snapshot.command_yaw = command_yaw;
      snapshot.command_pitch = command_pitch;
      snapshot.gimbal_yaw_deg = gimbal_yaw_deg;
      snapshot.gimbal_pitch_deg = gimbal_pitch_deg;
      snapshot.yaw_tracking_error_deg = yaw_tracking_error;
      snapshot.pitch_tracking_error_deg = pitch_tracking_error;
      snapshot.command_stable_s = command_stable_s;
      snapshot.prediction_error_deg = latest_prediction_error_deg;
      snapshot.prediction_error_m = latest_prediction_error_m;
      {
        std::lock_guard<std::mutex> lock(control_snapshot_mutex);
        control_snapshot = snapshot;
      }

      std::this_thread::sleep_for(7ms);
    }
  });

  constexpr char window_name[] = "Auto Drone Pipeline Diagnostics";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  MouseState mouse_state;
  cv::setMouseCallback(window_name, mouse_callback, &mouse_state);

  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point last_capture_timestamp;
  auto detection_window_start = std::chrono::steady_clock::now();
  uint64_t frame_id = 0;
  int detection_window_count = 0;
  double capture_fps = 0.0;
  double detection_fps = 0.0;
  bool static_test = false;
  std::deque<AngularSample> static_samples;
  std::map<std::string, uint64_t> failure_counts;
  int return_code = 0;

  while (!exiter.exit()) {
    camera.read(image, timestamp);
    if (
      last_capture_timestamp != std::chrono::steady_clock::time_point{} &&
      timestamp > last_capture_timestamp) {
      capture_fps = 1.0 / std::chrono::duration<double>(timestamp - last_capture_timestamp).count();
    }
    last_capture_timestamp = timestamp;

    std::optional<auto_drone::YOLOResult> result;
    try {
      result = yolo.detect_async(image, timestamp, frame_id++);
    } catch (const std::exception & error) {
      tools::logger()->error("[AutoDroneDiagnostics] Inference failed: {}", error.what());
      frame_queue.push(FramePacket{});
      return_code = 1;
      break;
    }
    if (!result) continue;

    image = std::move(result->frame);
    timestamp = result->timestamp;
    auto drones = std::move(result->drones);
    const auto now = std::chrono::steady_clock::now();
    ++detection_window_count;
    const double window_s = std::chrono::duration<double>(now - detection_window_start).count();
    if (window_s >= 1.0) {
      detection_fps = detection_window_count / window_s;
      detection_window_count = 0;
      detection_window_start = now;
    }

    FramePacket packet;
    packet.valid_frame = true;
    packet.frame_id = result->frame_id;
    packet.timestamp = timestamp;
    packet.capture_fps = capture_fps;
    packet.detection_fps = detection_fps;
    packet.detection_latency_ms =
      std::chrono::duration<double, std::milli>(now - timestamp).count();
    packet.image_width = image.cols;
    packet.image_height = image.rows;
    packet.static_test = static_test;

    packet.pose = gimbal.q_diagnostic(timestamp);
    solver.set_R_gimbal2world(packet.pose.q);
    auto targets = tracker.track(drones, timestamp);
    packet.tracker_state = tracker.state();
    if (!targets.empty()) packet.target = targets.front();

    if (!drones.empty()) {
      packet.observed_drone = drones.front();
      packet.solve = solver.diagnose(drones.front());
      if (!drones.front().point_confidences.empty()) {
        packet.minimum_keypoint_confidence = *std::min_element(
          drones.front().point_confidences.begin(), drones.front().point_confidences.end());
      }
    } else {
      packet.solve.principal_point = solver.principal_point();
    }

    if (static_test && packet.solve.pnp_success) {
      const auto ypd = tools::xyz2ypd(packet.solve.origin_world);
      static_samples.push_back({timestamp, ypd.x(), ypd.y()});
      while (!static_samples.empty() && timestamp - static_samples.front().timestamp > 2s) {
        static_samples.pop_front();
      }
      std::tie(packet.world_yaw_std_deg, packet.world_pitch_std_deg) =
        angular_standard_deviation(static_samples);
    }

    {
      std::lock_guard<std::mutex> lock(mouse_state.mutex);
      packet.laser_reference = mouse_state.laser_reference;
      packet.observed_laser_spot = mouse_state.observed_laser_spot;
    }
    if (packet.laser_reference && packet.solve.pnp_success) {
      packet.laser_target_error_deg = ray_angle_deg(
        solver.pixel_ray(*packet.laser_reference), solver.pixel_ray(packet.solve.origin_pixel));
      if (packet.observed_laser_spot) {
        packet.laser_spot_drift_deg = ray_angle_deg(
          solver.pixel_ray(*packet.laser_reference), solver.pixel_ray(*packet.observed_laser_spot));
      }
    }

    frame_queue.push(packet);

    ControlSnapshot control;
    {
      std::lock_guard<std::mutex> lock(control_snapshot_mutex);
      control = control_snapshot;
    }

    const cv::Point2f image_center(image.cols * 0.5F, image.rows * 0.5F);
    draw_cross(image, image_center, {225, 225, 225}, 20);
    draw_cross(image, solver.principal_point(), {230, 120, 40}, 18);
    tools::draw_text(image, "IMAGE", image_center + cv::Point2f(10, -8), {225, 225, 225}, 0.38, 1);
    tools::draw_text(
      image, "PRINCIPAL", solver.principal_point() + cv::Point2f(10, -8), {230, 120, 40}, 0.38, 1);
    if (packet.laser_reference) {
      draw_cross(image, *packet.laser_reference, {40, 40, 250}, 24);
      tools::draw_text(
        image, "LASER REF", *packet.laser_reference + cv::Point2f(10, -8), {40, 40, 250}, 0.4, 1);
    }
    if (packet.observed_laser_spot) {
      draw_cross(image, *packet.observed_laser_spot, {40, 150, 250}, 18);
      tools::draw_text(
        image, "LASER SPOT", *packet.observed_laser_spot + cv::Point2f(10, 16), {40, 150, 250}, 0.4,
        1);
    }

    if (packet.observed_drone) {
      const auto & drone = *packet.observed_drone;
      cv::rectangle(image, drone.box, {200, 255, 0}, 2, cv::LINE_AA);
      for (size_t i = 0; i < drone.points.size(); ++i) {
        cv::circle(image, drone.points[i], 5, {0, 230, 0}, cv::FILLED, cv::LINE_AA);
        const double confidence =
          i < drone.point_confidences.size() ? drone.point_confidences[i] : nan_value();
        tools::draw_text(
          image, fmt::format("{}:{:.2f}", i, confidence), drone.points[i] + cv::Point2f(7, -5),
          {0, 230, 0}, 0.42, 1);
      }
    }
    if (packet.solve.pnp_success) {
      draw_reprojection(image, packet.solve.reprojected_points, {255, 80, 255});
      draw_cross(image, packet.solve.keypoint_center, {0, 230, 230}, 20);
      draw_cross(image, packet.solve.origin_pixel, {255, 230, 40}, 24);
      draw_cross(image, packet.solve.center_pixel, {70, 230, 70}, 18);
      tools::draw_text(
        image, "RAW", packet.solve.keypoint_center + cv::Point2f(10, 16), {0, 230, 230}, 0.4, 1);
      tools::draw_text(
        image, "TVEC", packet.solve.origin_pixel + cv::Point2f(10, -8), {255, 230, 40}, 0.4, 1);
      tools::draw_text(
        image, "CENTER", packet.solve.center_pixel + cv::Point2f(10, 16), {70, 230, 70}, 0.4, 1);
    }
    if (control.valid && control.plan.plan_valid) {
      const auto predicted = control.plan.predicted_xyz;
      const std::vector<cv::Point3f> world_point{cv::Point3f(
        static_cast<float>(predicted.x()), static_cast<float>(predicted.y()),
        static_cast<float>(predicted.z()))};
      const auto pixels = solver.world2pixel(world_point);
      if (!pixels.empty()) {
        draw_cross(image, pixels.front(), {220, 60, 220}, 22);
        tools::draw_text(
          image, "PRED", pixels.front() + cv::Point2f(10, -8), {220, 60, 220}, 0.4, 1);
      }
    }

    std::vector<StageRow> rows;
    StageStatus keypoint_status = StageStatus::unknown;
    if (packet.observed_drone) {
      if (packet.observed_drone->points.size() != 8) {
        keypoint_status = StageStatus::failure;
      } else if (std::isfinite(packet.minimum_keypoint_confidence)) {
        keypoint_status = packet.minimum_keypoint_confidence >= 0.5    ? StageStatus::ok
                          : packet.minimum_keypoint_confidence >= 0.25 ? StageStatus::warning
                                                                       : StageStatus::failure;
      }
    }
    rows.push_back(
      {"KEYPOINTS",
       fmt::format(
         "n={} min={:.2f}", packet.observed_drone ? packet.observed_drone->points.size() : 0,
         packet.minimum_keypoint_confidence),
       keypoint_status});

    const double raw_origin_px =
      packet.solve.pnp_success ? cv::norm(packet.solve.keypoint_center - packet.solve.origin_pixel)
                               : nan_value();
    const double origin_center_px =
      packet.solve.pnp_success ? cv::norm(packet.solve.origin_pixel - packet.solve.center_pixel)
                               : nan_value();
    rows.push_back(
      {"AIM POINT", fmt::format("raw-o={:.1f}px o-c={:.1f}px", raw_origin_px, origin_center_px),
       packet.solve.pnp_success ? StageStatus::ok : StageStatus::unknown});

    StageStatus pnp_status = StageStatus::unknown;
    if (packet.solve.valid_input && !packet.solve.pnp_success) {
      pnp_status = StageStatus::failure;
    } else if (packet.solve.pnp_success) {
      pnp_status = lower_is_better(packet.solve.reprojection_rmse_px, 2.0, 5.0);
    }
    rows.push_back(
      {"PNP",
       fmt::format(
         "rmse={:.2f}px max={:.2f}px z={:.2f}m", packet.solve.reprojection_rmse_px,
         packet.solve.reprojection_max_px, packet.solve.origin_camera.z()),
       packet.solve.pnp_success && packet.solve.origin_camera.z() <= 0.0 ? StageStatus::failure
                                                                         : pnp_status});

    rows.push_back(
      {"CAMERA",
       fmt::format(
         "xyz={:.2f},{:.2f},{:.2f}", packet.solve.origin_camera.x(), packet.solve.origin_camera.y(),
         packet.solve.origin_camera.z()),
       packet.solve.pnp_success && packet.solve.origin_camera.allFinite() ? StageStatus::ok
                                                                          : StageStatus::unknown});

    rows.push_back(
      {"GIMBAL/IMU", fmt::format("roundtrip={:.3f}px", packet.solve.roundtrip_error_px),
       packet.solve.pnp_success ? lower_is_better(packet.solve.roundtrip_error_px, 1.0, 3.0)
                                : StageStatus::unknown});

    StageStatus pose_status = lower_is_better(std::abs(packet.pose.used_sample_age_ms), 2.0, 5.0);
    if (packet.pose.legacy_early_return) pose_status = StageStatus::failure;
    if (camera.timestamp_offset.count() != 0) pose_status = StageStatus::failure;
    rows.push_back(
      {"POSE TIME",
       fmt::format(
         "age={:.2f}ms span={:.2f}ms {} off={}us UNUSED", packet.pose.used_sample_age_ms,
         packet.pose.sample_span_ms, packet.pose.interpolated ? "INTERP" : "OLD",
         camera.timestamp_offset.count()),
       pose_status});

    const auto world_ypd = tools::xyz2ypd(packet.solve.origin_world);
    const double world_max_std = std::max(packet.world_yaw_std_deg, packet.world_pitch_std_deg);
    rows.push_back(
      {"WORLD",
       fmt::format(
         "y={:.2f} p={:.2f} std={:.3f}/{:.3f}", rad_to_deg(world_ypd.x()),
         rad_to_deg(world_ypd.y()), packet.world_yaw_std_deg, packet.world_pitch_std_deg),
       static_test ? lower_is_better(world_max_std, 0.1, 0.3) : StageStatus::unknown});

    rows.push_back(
      {"PREDICT",
       fmt::format(
         "h={:.1f}ms review={:.3f}deg", control.plan.prediction_horizon_s * 1e3,
         control.prediction_error_deg),
       lower_is_better(control.prediction_error_deg, 0.2, 0.5)});

    rows.push_back(
      {"YAW/PITCH",
       fmt::format(
         "raw={:.2f}/{:.2f} cmd={:.2f}/{:.2f}", rad_to_deg(control.plan.input_yaw_pitch.x()),
         rad_to_deg(control.plan.input_yaw_pitch.y()), control.command_yaw, control.command_pitch),
       control.plan.plan_valid ? StageStatus::ok : StageStatus::unknown});

    const StageStatus protocol_status =
      control.target_present ? StageStatus::warning : StageStatus::failure;
    rows.push_back(
      {"PROTOCOL",
       fmt::format(
         "P,Y={:.2f},{:.2f} CRC={} flags=NO{}", control.send.serialized_pitch,
         control.send.serialized_yaw, control.send.crc_valid ? "OK" : "BAD",
         control.target_present ? "" : " LEGACY-RAD"),
       control.send.crc_valid ? protocol_status : StageStatus::failure});

    const double maximum_control_error = std::max(
      std::abs(control.yaw_tracking_error_deg), std::abs(control.pitch_tracking_error_deg));
    rows.push_back(
      {"LOWER CTRL",
       fmt::format(
         "err={:.2f}/{:.2f}deg stable={:.2f}s", control.yaw_tracking_error_deg,
         control.pitch_tracking_error_deg, control.command_stable_s),
       lower_is_better(maximum_control_error, 0.3, 1.0)});

    const double laser_metric = std::isfinite(packet.laser_spot_drift_deg)
                                  ? packet.laser_spot_drift_deg
                                  : packet.laser_target_error_deg;
    rows.push_back(
      {"LASER/MECH",
       fmt::format(
         "target={:.3f}deg drift={:.3f}deg", packet.laser_target_error_deg,
         packet.laser_spot_drift_deg),
       lower_is_better(laser_metric, 0.1, 0.3)});

    std::string first_failure;
    for (auto & row : rows) {
      if (row.status == StageStatus::failure) ++failure_counts[row.name];
      const auto count = failure_counts.find(row.name);
      if (count != failure_counts.end() && count->second > 0) {
        row.metric += fmt::format(" #{}", count->second);
      }
      if (row.status == StageStatus::failure) {
        if (first_failure.empty()) first_failure = row.name;
      }
    }

    cv::Mat displayed_image;
    cv::resize(image, displayed_image, {}, kDisplayScale, kDisplayScale, cv::INTER_AREA);
    const int content_image_height = displayed_image.rows;
    constexpr int minimum_dashboard_height = 620;
    if (displayed_image.rows < minimum_dashboard_height) {
      cv::copyMakeBorder(
        displayed_image, displayed_image, 0, minimum_dashboard_height - displayed_image.rows, 0, 0,
        cv::BORDER_CONSTANT, cv::Scalar(18, 20, 22));
    }
    cv::Mat panel =
      draw_panel(displayed_image.rows, rows, packet.frame_id, static_test, first_failure);
    cv::Mat dashboard;
    cv::hconcat(displayed_image, panel, dashboard);
    {
      std::lock_guard<std::mutex> lock(mouse_state.mutex);
      mouse_state.scale = kDisplayScale;
      mouse_state.displayed_image_width = displayed_image.cols;
      mouse_state.displayed_image_height = content_image_height;
    }
    cv::imshow(window_name, dashboard);

    const int key = cv::waitKey(1);
    if (key == 'q') break;
    if (key == 's') {
      static_test = !static_test;
      static_samples.clear();
    }
    if (key == 'r') {
      static_samples.clear();
      failure_counts.clear();
      reset_generation.fetch_add(1);
      std::lock_guard<std::mutex> lock(mouse_state.mutex);
      mouse_state.observed_laser_spot.reset();
    }
  }

  quit = true;
  if (control_thread.joinable()) control_thread.join();

  const auto state = gimbal.state();
  gimbal.drone_send(false, false, state.yaw / 57.3F, 0.0F, 0.0F, state.pitch / 57.3F, 0.0F, 0.0F);
  return return_code;
}
