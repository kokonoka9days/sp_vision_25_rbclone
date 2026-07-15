#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | ../configs/demo.yaml | yaml配置文件的路径}"
  "{enemy-color    | blue               | any、config、red或blue，验证目标颜色}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{horizon        | 0.030             | 未来位置预测时长，单位s}"
  "{horizons       |                   | 逗号分隔的多个预测时长，设置后覆盖horizon}"
  "{score-sigma    | 0.22              | 高斯评分的距离标准差，单位m}"
  "{score-mode     | tracking          | tracking横向跟随误差或euclidean三维误差}"
  "{min-score      | 95.0              | 通过所需的最低平均高斯分数，0表示禁用}"
  "{min-coverage   | 90.0              | 通过所需的最低匹配覆盖率，单位百分比}"
  "{min-tracking-coverage | 90.0        | 预测输出帧占全部输入帧的最低比例}"
  "{min-input-coverage | 90.0           | 实际处理帧占请求视频帧的最低比例}"
  "{min-matched    | 100               | 每个通过时域所需的最少匹配样本数}"
  "{min-baseline-gain | -0.1           | 相对保持基线最低增益，单位百分点}"
  "{max-interp-gap | 0.08              | 真值插值允许的最大观测间隔，单位s}"
  "{truth-smoothing-window | 0.30      | 原始PnP伪真值的鲁棒零相位平滑窗口，单位s}"
  "{warmup         | 0.5               | 每个连续目标段的预热时间，单位s}"
  "{fixed-camera   | false             | 纯视频模式：无IMU，使用单位旋转和视频帧率}"
  "{fps            | 0                 | 纯视频帧率覆盖，0表示读取视频元数据}"
  "{headless       | true              | 关闭Plotter和GUI以批量评测}"
  "{output o       | prediction_eval.csv | 逐样本评估CSV路径}"
  "{@input-path    | ../assets/demo/demo  | avi和txt文件的路径}";

namespace
{

constexpr double kTrackerContinuityGap = 0.1;

struct PredictionSample
{
  int frame;
  int segment;
  auto_aim::Color color;
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  double prediction_time;
  double target_time;
  Eigen::Vector3d cv_center;
  Eigen::Vector3d ca_center;
  Eigen::Vector3d fused_center;
  Eigen::Vector3d hold_center;
  bool ca_ready;
  double w_cv;
};

struct ObservationSample
{
  int frame;
  int segment;
  auto_aim::Color color;
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  double time;
  Eigen::Vector3d raw_center;
  Eigen::Vector3d center;
};

struct MetricAccumulator
{
  std::vector<double> errors;
  double squared_error_sum = 0.0;
  double score_sum = 0.0;

  void add(double error, double score_sigma)
  {
    const double normalized_error = error / score_sigma;
    errors.push_back(error);
    squared_error_sum += error * error;
    score_sum += 100.0 * std::exp(-0.5 * normalized_error * normalized_error);
  }

  double mean_score() const
  {
    return errors.empty() ? std::numeric_limits<double>::quiet_NaN()
                          : score_sum / static_cast<double>(errors.size());
  }

  double rmse() const
  {
    return errors.empty()
             ? std::numeric_limits<double>::quiet_NaN()
             : std::sqrt(squared_error_sum / static_cast<double>(errors.size()));
  }
};

struct HorizonMetrics
{
  std::size_t prediction_count = 0;
  std::size_t matched_count = 0;
  std::size_t ca_matched_count = 0;
  std::size_t hit_at_01_count = 0;
  MetricAccumulator cv;
  MetricAccumulator ca;
  MetricAccumulator fused;
  MetricAccumulator hold;
};

using RawTargetKey = std::tuple<int, int, int>;

struct RawCandidate
{
  auto_aim::Color color;
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  double image_distance;
  Eigen::Vector3d center;
};

struct RawTrackState
{
  int segment = -1;
  double last_time = std::numeric_limits<double>::quiet_NaN();
};

RawTargetKey raw_target_key(
  auto_aim::Color color, auto_aim::ArmorName name, auto_aim::ArmorType armor_type)
{
  return {
    static_cast<int>(color), static_cast<int>(name), static_cast<int>(armor_type)};
}

double nominal_radius(auto_aim::ArmorName name)
{
  if (name == auto_aim::ArmorName::outpost) return 0.2765;
  if (name == auto_aim::ArmorName::base) return 0.3205;
  return 0.2;
}

Eigen::Vector3d nominal_center(const auto_aim::Armor & armor)
{
  Eigen::Vector3d center = armor.xyz_in_world;
  const double radius = nominal_radius(armor.name);
  center.x() += radius * std::cos(armor.ypr_in_world[0]);
  center.y() += radius * std::sin(armor.ypr_in_world[0]);
  return center;
}

void smooth_raw_observations(std::vector<ObservationSample> & observations, double window)
{
  std::map<int, std::vector<std::size_t>> segments;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    segments[observations[i].segment].push_back(i);
  }

  std::vector<Eigen::Vector3d> smoothed;
  smoothed.reserve(observations.size());
  for (const auto & observation : observations) smoothed.push_back(observation.raw_center);

  const double half_window = 0.5 * window;
  for (const auto & [unused_segment, indices] : segments) {
    for (const std::size_t center_index : indices) {
      const double center_time = observations[center_index].time;
      std::vector<std::pair<std::size_t, double>> neighbors;
      for (const std::size_t index : indices) {
        const double dt = observations[index].time - center_time;
        const double normalized_time = std::abs(dt) / half_window;
        if (normalized_time >= 1.0) continue;
        const double one_minus_cube = 1.0 - normalized_time * normalized_time * normalized_time;
        neighbors.emplace_back(index, one_minus_cube * one_minus_cube * one_minus_cube);
      }
      if (neighbors.size() < 5) continue;

      std::vector<double> robust_weights(neighbors.size(), 1.0);
      Eigen::Matrix<double, 3, 3> coefficients = Eigen::Matrix3d::Zero();
      bool fit_valid = false;
      for (int iteration = 0; iteration < 3; ++iteration) {
        Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d rhs = Eigen::Matrix3d::Zero();
        for (std::size_t j = 0; j < neighbors.size(); ++j) {
          const auto [index, kernel_weight] = neighbors[j];
          const double dt = observations[index].time - center_time;
          const Eigen::Vector3d basis(1.0, dt, dt * dt);
          const double weight = kernel_weight * robust_weights[j];
          normal.noalias() += weight * basis * basis.transpose();
          rhs.noalias() += weight * basis * observations[index].raw_center.transpose();
        }

        const Eigen::LDLT<Eigen::Matrix3d> ldlt(normal);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) break;
        coefficients = ldlt.solve(rhs);
        if (!coefficients.allFinite()) break;
        fit_valid = true;

        std::vector<double> residuals;
        residuals.reserve(neighbors.size());
        for (const auto & [index, unused_weight] : neighbors) {
          const double dt = observations[index].time - center_time;
          const Eigen::Vector3d basis(1.0, dt, dt * dt);
          residuals.push_back(
            (coefficients.transpose() * basis - observations[index].raw_center).norm());
        }
        std::vector<double> sorted_residuals = residuals;
        std::sort(sorted_residuals.begin(), sorted_residuals.end());
        const double scale = 1.4826 * sorted_residuals[sorted_residuals.size() / 2] + 1e-6;
        for (std::size_t j = 0; j < residuals.size(); ++j) {
          const double u = residuals[j] / (4.685 * scale);
          if (u >= 1.0) {
            robust_weights[j] = 0.0;
          } else {
            const double one_minus_square = 1.0 - u * u;
            robust_weights[j] = one_minus_square * one_minus_square;
          }
        }
      }
      if (fit_valid) smoothed[center_index] = coefficients.row(0).transpose();
    }
  }

  for (std::size_t i = 0; i < observations.size(); ++i) observations[i].center = smoothed[i];
}

bool parse_horizons(const std::string & value, std::vector<double> & horizons)
{
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    std::stringstream token_stream(token);
    double horizon;
    std::string trailing;
    if (!(token_stream >> horizon) || (token_stream >> trailing) || !std::isfinite(horizon) ||
        horizon <= 0.0) {
      return false;
    }
    horizons.push_back(horizon);
  }
  if (horizons.empty()) return false;

  std::sort(horizons.begin(), horizons.end());
  horizons.erase(
    std::unique(
      horizons.begin(), horizons.end(),
      [](double lhs, double rhs) { return std::abs(lhs - rhs) < 1e-9; }),
    horizons.end());
  return true;
}

double percentile(const std::vector<double> & sorted_values, double p)
{
  if (sorted_values.empty()) return std::numeric_limits<double>::quiet_NaN();

  const double position = p * static_cast<double>(sorted_values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - alpha) + sorted_values[upper] * alpha;
}

double gaussian_score(double error, double score_sigma)
{
  const double normalized_error = error / score_sigma;
  return 100.0 * std::exp(-0.5 * normalized_error * normalized_error);
}

double scored_position_error(
  const Eigen::Vector3d & prediction, const Eigen::Vector3d & truth,
  const std::string & score_mode)
{
  const Eigen::Vector3d error = prediction - truth;
  if (score_mode == "euclidean") return error.norm();
  const double truth_range = truth.norm();
  if (truth_range < 1e-9) return error.norm();
  const Eigen::Vector3d line_of_sight = truth / truth_range;
  return (error - error.dot(line_of_sight) * line_of_sight).norm();
}

}  // namespace

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto enemy_color = cli.get<std::string>("enemy-color");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto horizon = cli.get<double>("horizon");
  auto horizons_argument = cli.get<std::string>("horizons");
  auto score_sigma = cli.get<double>("score-sigma");
  auto score_mode = cli.get<std::string>("score-mode");
  auto min_score = cli.get<double>("min-score");
  auto min_coverage = cli.get<double>("min-coverage");
  auto min_tracking_coverage = cli.get<double>("min-tracking-coverage");
  auto min_input_coverage = cli.get<double>("min-input-coverage");
  auto min_matched = cli.get<int>("min-matched");
  auto min_baseline_gain = cli.get<double>("min-baseline-gain");
  auto max_interp_gap = cli.get<double>("max-interp-gap");
  auto truth_smoothing_window = cli.get<double>("truth-smoothing-window");
  auto warmup = cli.get<double>("warmup");
  auto fixed_camera = cli.get<bool>("fixed-camera");
  auto fps_override = cli.get<double>("fps");
  auto headless = cli.get<bool>("headless");
  auto output_path = cli.get<std::string>("output");

  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }
  std::vector<double> horizons;
  if (horizons_argument.empty()) {
    horizons.push_back(horizon);
  } else if (!parse_horizons(horizons_argument, horizons)) {
    fmt::print(stderr, "Invalid --horizons value; expected positive comma-separated seconds.\n");
    return 2;
  }
  if (
    start_index < 0 || (end_index > 0 && end_index < start_index) || !std::isfinite(horizon) ||
    horizon <= 0.0 || !std::isfinite(score_sigma) || score_sigma <= 0.0 ||
    !std::isfinite(min_score) || min_score < 0.0 || min_score > 100.0 ||
    !std::isfinite(min_coverage) || min_coverage < 0.0 || min_coverage > 100.0 ||
    !std::isfinite(min_tracking_coverage) || min_tracking_coverage < 0.0 ||
    min_tracking_coverage > 100.0 || !std::isfinite(min_input_coverage) ||
    min_input_coverage < 0.0 || min_input_coverage > 100.0 || min_matched < 1 ||
    !std::isfinite(min_baseline_gain) || !std::isfinite(max_interp_gap) ||
    max_interp_gap <= 0.0 || !std::isfinite(truth_smoothing_window) ||
    truth_smoothing_window <= 0.0 || !std::isfinite(warmup) || warmup < 0.0 ||
    !std::isfinite(fps_override) || fps_override < 0.0 || output_path.empty() ||
    (score_mode != "tracking" && score_mode != "euclidean") ||
    (enemy_color != "any" && enemy_color != "config" && enemy_color != "red" &&
     enemy_color != "blue")) {
    fmt::print(stderr, "Invalid evaluation arguments. Use --help for valid ranges.\n");
    return 2;
  }

  // 诊断输出：CA vs CV
  const std::string diag_path = output_path + ".diag.csv";
  std::ofstream diag(diag_path);
  if (!diag) {
    fmt::print(stderr, "Failed to open diagnostic output: {}\n", diag_path);
    return 1;
  }
  diag << "frame,name,"
       << "cv_x,cv_y,cv_vx,cv_vy,cv_vyaw,"
       << "ca_x,ca_y,ca_vx,ca_vy,ca_ax,ca_ay,"
       << "fused_x,fused_y,w_cv,radius,nis_failure_rate\n";

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);
  if (!video.isOpened()) {
    fmt::print(stderr, "Failed to open video: {}\n", video_path);
    return 1;
  }
  if (!fixed_camera && !text) {
    fmt::print(stderr, "Failed to open timestamp file: {}\n", text_path);
    return 1;
  }
  const double video_fps = fps_override > 0.0 ? fps_override : video.get(cv::CAP_PROP_FPS);
  const double video_frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
  int expected_input_frames = 0;
  if (std::isfinite(video_frame_count) && video_frame_count > 0.0) {
    const int total_frames = static_cast<int>(std::llround(video_frame_count));
    const int last_requested_frame =
      end_index > 0 ? std::min(end_index, total_frames - 1) : total_frames - 1;
    expected_input_frames = std::max(0, last_requested_frame - start_index + 1);
  }
  if (expected_input_frames == 0) {
    cv::VideoCapture frame_counter(video_path);
    int total_frames = 0;
    while (frame_counter.grab()) total_frames++;
    const int last_requested_frame =
      end_index > 0 ? std::min(end_index, total_frames - 1) : total_frames - 1;
    expected_input_frames = std::max(0, last_requested_frame - start_index + 1);
  }
  if (fixed_camera && (!std::isfinite(video_fps) || video_fps <= 0.0)) {
    fmt::print(stderr, "Video FPS is unavailable; pass --fps with a positive value.\n");
    return 1;
  }

  auto_aim::YOLO yolo(config_path);
  auto_aim::Detector traditional(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, &solver);
  if (enemy_color == "red") tracker.set_enemy_color(auto_aim::Color::red);
  if (enemy_color == "blue") tracker.set_enemy_color(auto_aim::Color::blue);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img, drawing;
  auto t0 = std::chrono::steady_clock::now();

  io::Command last_command{false, false, 0, 0};

  std::vector<PredictionSample> predictions;
  std::vector<ObservationSample> observations;
  std::map<RawTargetKey, RawTrackState> raw_track_states;
  int next_raw_segment = 0;
  int segment = -1;
  int processed_frames = 0;
  bool target_active = false;
  double segment_start_time = 0.0;
  double previous_input_time = std::numeric_limits<double>::quiet_NaN();
  bool timestamp_integrity_error = false;
  auto_aim::ArmorName active_name{};
  auto_aim::ArmorType active_armor_type{};

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; !fixed_camera && i < start_index; i++) {
    double t, w, x, y, z;
    if (!(text >> t >> w >> x >> y >> z)) {
      fmt::print(stderr, "Timestamp file ended while seeking to frame {}.\n", start_index);
      return 1;
    }
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;
    // auto inshow_start = std::chrono::steady_clock::now();
    video.read(img);
    if (img.empty()) break;

    double t;
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (fixed_camera) {
      t = static_cast<double>(frame_count) / video_fps;
    } else {
      if (!(text >> t >> w >> x >> y >> z)) {
        tools::logger()->warn(
          "Timestamp file ended before video at frame {}. Evaluation stops at the last aligned frame.",
          frame_count);
        break;
      }
    }
    if (std::isfinite(previous_input_time) && t <= previous_input_time) {
      tools::logger()->error(
        "Non-monotonic timestamp at frame {}: {:.6f} <= {:.6f}. Evaluation stops.", frame_count,
        t, previous_input_time);
      timestamp_integrity_error = true;
      break;
    }
    const bool input_discontinuity =
      std::isfinite(previous_input_time) && t - previous_input_time > kTrackerContinuityGap;
    previous_input_time = t;
    processed_frames++;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    // auto traditional_start = std::chrono::steady_clock::now();
    // auto armors = traditional.detect(img, frame_count);

    // Build EKF-independent pseudo ground truth from every raw detection before Tracker mutates,
    // filters, or innovation-gates the list. One candidate per identity is retained per frame.
    std::map<RawTargetKey, RawCandidate> raw_candidates;
    const cv::Point2f image_center(
      static_cast<float>(img.cols) * 0.5F, static_cast<float>(img.rows) * 0.5F);
    for (auto armor : armors) {
      if (enemy_color == "red" && armor.color != auto_aim::Color::red) continue;
      if (enemy_color == "blue" && armor.color != auto_aim::Color::blue) continue;
      if (!solver.solve(armor)) continue;

      const Eigen::Vector3d center = nominal_center(armor);
      if (!center.allFinite()) continue;
      const double image_distance = cv::norm(armor.center - image_center);
      const auto key = raw_target_key(armor.color, armor.name, armor.type);
      const RawCandidate candidate{
        armor.color, armor.name, armor.type, image_distance, center};
      const auto existing = raw_candidates.find(key);
      if (existing == raw_candidates.end()) {
        raw_candidates.emplace(key, candidate);
      } else if (image_distance < existing->second.image_distance) {
        existing->second = candidate;
      }
    }

    for (const auto & [key, candidate] : raw_candidates) {
      auto & state = raw_track_states[key];
      if (!std::isfinite(state.last_time) || t - state.last_time > kTrackerContinuityGap) {
        state.segment = next_raw_segment++;
      }
      state.last_time = t;
      observations.push_back(
        {frame_count, state.segment, candidate.color, candidate.name, candidate.armor_type, t,
         candidate.center, candidate.center});
    }

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.test_track(armors, timestamp, true, enemy_color != "any");

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    // tools::logger()->info(
    //   "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
    //   tools::delta_time(tracker_start, yolo_start) * 1e3,
    //   tools::delta_time(aimer_start, tracker_start) * 1e3,
    //   tools::delta_time(finish, aimer_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion gimbal_q = {w, x, y, z};
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    Eigen::Quaternion q{w, x, y, z};
    auto yaw = tools::eulers(q, 2, 1, 0)[0];
    data["gimbal_yaw"] = yaw * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      const bool target_changed =
        !target_active || target.name != active_name || target.armor_type != active_armor_type;
      if (input_discontinuity || target_changed) {
        segment++;
        segment_start_time = t;
      }
      target_active = true;
      active_name = target.name;
      active_armor_type = target.armor_type;

      if (t - segment_start_time >= warmup) {
        const Eigen::Vector3d hold_center = target.fused_center();
        for (const double prediction_horizon : horizons) {
          auto predicted_target = target;
          predicted_target.predict(prediction_horizon);
          const Eigen::Vector3d cv_center = predicted_target.cv_center();
          const Eigen::Vector3d fused_center = predicted_target.fused_center();
          const bool ca_ready = predicted_target.ca_ekf_ready();
          Eigen::Vector3d ca_center =
            Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
          if (ca_ready) {
            const Eigen::VectorXd ca_x = predicted_target.ca_ekf_x();
            ca_center = {ca_x[0], ca_x[3], ca_x[6]};
          }
          if (cv_center.allFinite() && fused_center.allFinite()) {
            predictions.push_back(
              {frame_count, segment, target.color, target.name, target.armor_type, t,
               t + prediction_horizon, cv_center, ca_center, fused_center,
               hold_center, ca_ready && ca_center.allFinite(), predicted_target.get_w_cv()});
          }
        }
      }

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        Eigen::Vector4d aim_xyza = aim_point.xyza;
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
      }

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");

      // === CA vs CV 融合诊断 ===
      if (target.ca_ekf_ready()) {
        Eigen::VectorXd cv_x = target.ekf_x();
        Eigen::VectorXd ca_x = target.ca_ekf_x();
        double w_cv = target.get_w_cv();
        Eigen::Vector3d center_CV(cv_x[0], cv_x[2], cv_x[4]);
        Eigen::Vector3d center_CA(ca_x[0], ca_x[3], ca_x[6]);
        Eigen::Vector3d center_fused = target.fused_center();

        diag << std::fixed << std::setprecision(4)
             << frame_count << ","
             << static_cast<int>(target.name) << ","
             << center_CV.x() << "," << center_CV.y() << ","
             << cv_x[1] << "," << cv_x[3] << "," << cv_x[7] << ","
             << center_CA.x() << "," << center_CA.y() << ","
             << ca_x[1] << "," << ca_x[4] << ","
             << ca_x[2] << "," << ca_x[5] << ","
             << center_fused.x() << "," << center_fused.y() << ","
             << w_cv << "," << cv_x[8] << ","
             << target.ekf().data.at("recent_nis_failures") << "\n";
      }
    } else {
      target_active = false;
    }

    if (!headless) {
      plotter.plot(data);

      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
      auto key = cv::waitKey(30);
      if (key == 'q') break;
      while (key == ' ') {
        int y = cv::waitKey(30);
        if (y == 'q') break;
      }
    }

    //  tools::logger()->info(
    //     "imshow : {:.1f}ms",  tools::delta_time(std::chrono::steady_clock::now(), inshow_start) * 1e3);
  }

  smooth_raw_observations(observations, truth_smoothing_window);

  const std::string observations_path = output_path + ".observations.csv";
  std::ofstream observation_output(observations_path);
  if (!observation_output) {
    fmt::print(stderr, "Failed to open observation output: {}\n", observations_path);
    return 1;
  }
  observation_output
    << "frame,raw_segment,time_s,color,name,armor_type,raw_x_m,raw_y_m,raw_z_m,"
       "truth_x_m,truth_y_m,truth_z_m\n";
  observation_output << std::fixed << std::setprecision(9);
  for (const auto & observation : observations) {
    observation_output << observation.frame << ',' << observation.segment << ',' << observation.time
                       << ',' << static_cast<int>(observation.color) << ','
                       << static_cast<int>(observation.name) << ','
                       << static_cast<int>(observation.armor_type) << ','
                       << observation.raw_center.x() << ',' << observation.raw_center.y() << ','
                       << observation.raw_center.z() << ',' << observation.center.x() << ','
                       << observation.center.y() << ',' << observation.center.z() << '\n';
  }

  std::ofstream evaluation(output_path);
  if (!evaluation) {
    fmt::print(stderr, "Failed to open evaluation output: {}\n", output_path);
    return 1;
  }
  evaluation << "prediction_frame,tracker_segment,color,name,armor_type,prediction_time_s,target_time_s,horizon_s,"
             << "matched,status,observation_before_frame,observation_after_frame,"
             << "observation_before_time_s,observation_after_time_s,interpolation_alpha,"
             << "truth_raw_segment,source_w_cv,prediction_cv_x_m,prediction_cv_y_m,prediction_cv_z_m,"
             << "prediction_ca_x_m,prediction_ca_y_m,prediction_ca_z_m,"
             << "prediction_fused_x_m,prediction_fused_y_m,prediction_fused_z_m,"
             << "truth_x_m,truth_y_m,truth_z_m,"
             << "hold_x_m,hold_y_m,hold_z_m,hold_euclidean_error_m,hold_scored_error_m,hold_score_pct,"
             << "cv_error_x_m,cv_error_y_m,cv_error_z_m,cv_euclidean_error_m,cv_scored_error_m,cv_score_pct,"
             << "ca_error_x_m,ca_error_y_m,ca_error_z_m,ca_euclidean_error_m,ca_scored_error_m,ca_score_pct,"
             << "fused_error_x_m,fused_error_y_m,fused_error_z_m,fused_euclidean_error_m,fused_scored_error_m,"
             << "fused_score_pct\n";
  evaluation << std::fixed << std::setprecision(9);

  MetricAccumulator cv_metrics;
  MetricAccumulator ca_metrics;
  MetricAccumulator fused_metrics;
  MetricAccumulator hold_metrics;
  MetricAccumulator fused_euclidean_metrics;
  MetricAccumulator hold_euclidean_metrics;
  std::map<RawTargetKey, std::vector<const ObservationSample *>> observations_by_target;
  std::map<int, std::pair<double, double>> raw_segment_bounds;
  for (const auto & observation : observations) {
    observations_by_target[raw_target_key(
      observation.color, observation.name, observation.armor_type)]
      .push_back(&observation);
    const auto bounds = raw_segment_bounds.find(observation.segment);
    if (bounds == raw_segment_bounds.end()) {
      raw_segment_bounds.emplace(
        observation.segment, std::make_pair(observation.time, observation.time));
    } else {
      bounds->second.first = std::min(bounds->second.first, observation.time);
      bounds->second.second = std::max(bounds->second.second, observation.time);
    }
  }
  std::map<double, HorizonMetrics> horizon_metrics;
  for (const double value : horizons) horizon_metrics[value] = {};
  std::size_t hit_at_01_count = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  for (const auto & prediction : predictions) {
    const double prediction_horizon = prediction.target_time - prediction.prediction_time;
    auto & per_horizon = horizon_metrics.lower_bound(prediction_horizon - 1e-9)->second;
    per_horizon.prediction_count++;

    const ObservationSample * before = nullptr;
    const ObservationSample * after = nullptr;
    const ObservationSample * causal_observation = nullptr;
    Eigen::Vector3d truth = Eigen::Vector3d::Constant(nan);
    double alpha = nan;
    std::string status;

    constexpr double exact_time_tolerance = 1e-9;
    const auto group_it = observations_by_target.find(
      raw_target_key(prediction.color, prediction.name, prediction.armor_type));
    if (group_it == observations_by_target.end()) {
      status = "no_raw_target_identity";
    } else {
      const auto & target_observations = group_it->second;
      const auto upper = std::lower_bound(
        target_observations.begin(), target_observations.end(), prediction.target_time,
        [](const ObservationSample * observation, double time) {
          return observation->time < time;
        });
      const auto causal_upper = std::upper_bound(
        target_observations.begin(), target_observations.end(), prediction.prediction_time,
        [](double time, const ObservationSample * observation) {
          return time < observation->time;
        });
      if (causal_upper != target_observations.begin()) causal_observation = *(causal_upper - 1);
      if (
        upper != target_observations.end() &&
        std::abs((*upper)->time - prediction.target_time) <= exact_time_tolerance &&
        (*upper)->time > prediction.prediction_time + exact_time_tolerance) {
        before = *upper;
        after = *upper;
        truth = (*upper)->center;
        alpha = 0.0;
        status = "matched_exact_raw";
      } else if (upper == target_observations.end()) {
        status = "no_future_raw_observation";
      } else if (upper == target_observations.begin()) {
        status = "no_previous_raw_observation";
      } else {
        const ObservationSample * lower = *(upper - 1);
        const ObservationSample * upper_observation = *upper;
        if (lower->time <= prediction.prediction_time + exact_time_tolerance) {
          status = "truth_bracket_not_strictly_future";
        } else if (lower->segment != upper_observation->segment) {
          status = "raw_segment_mismatch";
        } else if (upper_observation->time - lower->time > max_interp_gap) {
          status = "interpolation_gap_too_large";
        } else {
          before = lower;
          after = upper_observation;
          alpha =
            (prediction.target_time - lower->time) / (upper_observation->time - lower->time);
          truth = (1.0 - alpha) * lower->center + alpha * upper_observation->center;
          status = "matched_interpolated_raw";
        }
      }
    }

    if (before != nullptr) {
      const auto bounds = raw_segment_bounds.at(before->segment);
      const double half_smoothing_window = 0.5 * truth_smoothing_window;
      if (
        prediction.target_time - bounds.first < half_smoothing_window ||
        bounds.second - prediction.target_time < half_smoothing_window) {
        before = nullptr;
        after = nullptr;
        truth.setConstant(nan);
        status = "truth_smoothing_window_incomplete";
      } else if (
        causal_observation == nullptr || causal_observation->segment != before->segment ||
        prediction.prediction_time - causal_observation->time > max_interp_gap) {
        before = nullptr;
        after = nullptr;
        truth.setConstant(nan);
        status = "no_causal_raw_anchor";
      }
    }

    const bool matched = before != nullptr && after != nullptr && truth.allFinite();
    Eigen::Vector3d cv_error = Eigen::Vector3d::Constant(nan);
    Eigen::Vector3d ca_error = Eigen::Vector3d::Constant(nan);
    Eigen::Vector3d fused_error = Eigen::Vector3d::Constant(nan);
    Eigen::Vector3d hold_center = Eigen::Vector3d::Constant(nan);
    double cv_distance_error = nan;
    double ca_distance_error = nan;
    double fused_distance_error = nan;
    double hold_distance_error = nan;
    double cv_euclidean_error = nan;
    double ca_euclidean_error = nan;
    double fused_euclidean_error = nan;
    double hold_euclidean_error = nan;
    double cv_score = nan;
    double ca_score = nan;
    double fused_score = nan;
    double hold_score = nan;
    if (matched) {
      cv_error = prediction.cv_center - truth;
      fused_error = prediction.fused_center - truth;
      cv_euclidean_error = cv_error.norm();
      fused_euclidean_error = fused_error.norm();
      cv_distance_error =
        scored_position_error(prediction.cv_center, truth, score_mode);
      fused_distance_error =
        scored_position_error(prediction.fused_center, truth, score_mode);
      cv_score = gaussian_score(cv_distance_error, score_sigma);
      fused_score = gaussian_score(fused_distance_error, score_sigma);
      cv_metrics.add(cv_distance_error, score_sigma);
      fused_metrics.add(fused_distance_error, score_sigma);
      fused_euclidean_metrics.add(fused_euclidean_error, score_sigma);
      per_horizon.matched_count++;
      per_horizon.cv.add(cv_distance_error, score_sigma);
      per_horizon.fused.add(fused_distance_error, score_sigma);

      if (prediction.hold_center.allFinite()) {
        hold_center = prediction.hold_center;
        hold_euclidean_error = (hold_center - truth).norm();
        hold_distance_error = scored_position_error(hold_center, truth, score_mode);
        hold_score = gaussian_score(hold_distance_error, score_sigma);
        hold_metrics.add(hold_distance_error, score_sigma);
        hold_euclidean_metrics.add(hold_euclidean_error, score_sigma);
        per_horizon.hold.add(hold_distance_error, score_sigma);
      }

      if (prediction.ca_ready) {
        ca_error = prediction.ca_center - truth;
        ca_euclidean_error = ca_error.norm();
        ca_distance_error =
          scored_position_error(prediction.ca_center, truth, score_mode);
        ca_score = gaussian_score(ca_distance_error, score_sigma);
        ca_metrics.add(ca_distance_error, score_sigma);
        per_horizon.ca_matched_count++;
        per_horizon.ca.add(ca_distance_error, score_sigma);
      }
      if (fused_distance_error <= 0.1) {
        hit_at_01_count++;
        per_horizon.hit_at_01_count++;
      }
    }

    evaluation << prediction.frame << ',' << prediction.segment << ','
               << static_cast<int>(prediction.color) << ','
               << static_cast<int>(prediction.name) << ','
               << static_cast<int>(prediction.armor_type) << ',' << prediction.prediction_time << ','
               << prediction.target_time << ',' << prediction_horizon << ',' << (matched ? 1 : 0)
               << ',' << status << ','
               << (before ? before->frame : -1) << ',' << (after ? after->frame : -1) << ','
               << (before ? before->time : nan) << ',' << (after ? after->time : nan) << ',' << alpha
               << ',' << (before ? before->segment : -1) << ',' << prediction.w_cv << ','
               << prediction.cv_center.x() << ','
               << prediction.cv_center.y() << ',' << prediction.cv_center.z() << ','
               << prediction.ca_center.x() << ',' << prediction.ca_center.y() << ','
               << prediction.ca_center.z() << ',' << prediction.fused_center.x() << ','
               << prediction.fused_center.y() << ',' << prediction.fused_center.z() << ',' << truth.x()
               << ',' << truth.y() << ',' << truth.z() << ',' << hold_center.x() << ','
               << hold_center.y() << ',' << hold_center.z() << ',' << hold_euclidean_error << ','
               << hold_distance_error << ',' << hold_score << ',' << cv_error.x() << ','
               << cv_error.y() << ',' << cv_error.z() << ',' << cv_euclidean_error << ','
               << cv_distance_error << ',' << cv_score << ','
               << ca_error.x() << ',' << ca_error.y() << ',' << ca_error.z() << ','
               << ca_euclidean_error << ',' << ca_distance_error << ',' << ca_score << ','
               << fused_error.x() << ',' << fused_error.y() << ',' << fused_error.z() << ','
               << fused_euclidean_error << ',' << fused_distance_error << ',' << fused_score << '\n';
  }

  std::sort(cv_metrics.errors.begin(), cv_metrics.errors.end());
  std::sort(ca_metrics.errors.begin(), ca_metrics.errors.end());
  std::sort(fused_metrics.errors.begin(), fused_metrics.errors.end());
  std::sort(hold_metrics.errors.begin(), hold_metrics.errors.end());
  std::sort(fused_euclidean_metrics.errors.begin(), fused_euclidean_metrics.errors.end());
  std::sort(hold_euclidean_metrics.errors.begin(), hold_euclidean_metrics.errors.end());
  for (auto & [unused_horizon, metrics] : horizon_metrics) {
    std::sort(metrics.cv.errors.begin(), metrics.cv.errors.end());
    std::sort(metrics.ca.errors.begin(), metrics.ca.errors.end());
    std::sort(metrics.fused.errors.begin(), metrics.fused.errors.end());
    std::sort(metrics.hold.errors.begin(), metrics.hold.errors.end());
  }
  const auto matched_count = fused_metrics.errors.size();
  const double coverage = predictions.empty()
                            ? nan
                            : 100.0 * static_cast<double>(matched_count) / predictions.size();
  const double median_error = percentile(fused_metrics.errors, 0.5);
  const double hit_at_01 = matched_count == 0
                             ? nan
                             : 100.0 * static_cast<double>(hit_at_01_count) / matched_count;
  const double baseline_gain = fused_metrics.mean_score() - hold_metrics.mean_score();
  const double prediction_frame_count = horizons.empty()
                                          ? 0.0
                                          : static_cast<double>(predictions.size()) /
                                              static_cast<double>(horizons.size());
  const double tracking_output_coverage =
    processed_frames == 0 ? nan : 100.0 * prediction_frame_count / processed_frames;
  const double input_coverage =
    expected_input_frames > 0
      ? 100.0 * static_cast<double>(processed_frames) / expected_input_frames
      : 100.0;

  fmt::print(
    "\nPrediction evaluation summary\n"
    "truth_source=robust_zero_phase_raw_pnp_trajectory\n"
    "truth_smoothing_window_s={:.3f}\n"
    "zero_phase_truth_uses_offline_context=true\n"
    "score_mode={}\n"
    "score_sigma_m={:.3f}\n"
    "future_raw_observation_bracket=true\n"
    "output={}\n"
    "observations_output={}\n"
    "processed_frames={}\n"
    "expected_input_frames={}\n"
    "input_coverage_pct={:.3f}\n"
    "segments={}\n"
    "raw_segments={}\n"
    "observation_count={}\n"
    "prediction_count={}\n"
    "matched_count={}\n"
    "coverage_pct={:.3f}\n"
    "tracking_output_coverage_pct={:.3f}\n"
    "mean_score_pct={:.3f}\n"
    "scored_rmse_m={:.6f}\n"
    "scored_median_error_m={:.6f}\n"
    "scored_p95_error_m={:.6f}\n"
    "hit_at_0.1m_pct={:.3f}\n"
    "hold_matched_count={}\n"
    "hold_mean_score_pct={:.3f}\n"
    "hold_scored_rmse_m={:.6f}\n"
    "hold_scored_p95_error_m={:.6f}\n"
    "hold_euclidean_rmse_m={:.6f}\n"
    "hold_euclidean_p95_error_m={:.6f}\n"
    "cv_mean_score_pct={:.3f}\n"
    "cv_rmse_m={:.6f}\n"
    "cv_p95_error_m={:.6f}\n"
    "ca_matched_count={}\n"
    "ca_mean_score_pct={:.3f}\n"
    "ca_rmse_m={:.6f}\n"
    "ca_p95_error_m={:.6f}\n"
    "fused_mean_score_pct={:.3f}\n"
    "fused_scored_rmse_m={:.6f}\n"
    "fused_scored_p95_error_m={:.6f}\n"
    "fused_euclidean_rmse_m={:.6f}\n"
    "fused_euclidean_p95_error_m={:.6f}\n"
    "fused_vs_hold_score_gain_pct={:.3f}\n",
    truth_smoothing_window, score_mode, score_sigma, output_path, observations_path,
    processed_frames, expected_input_frames, input_coverage, segment + 1, next_raw_segment,
    observations.size(),
    predictions.size(), matched_count, coverage, tracking_output_coverage,
    fused_metrics.mean_score(), fused_metrics.rmse(),
    median_error, percentile(fused_metrics.errors, 0.95), hit_at_01, hold_metrics.errors.size(),
    hold_metrics.mean_score(), hold_metrics.rmse(), percentile(hold_metrics.errors, 0.95),
    hold_euclidean_metrics.rmse(), percentile(hold_euclidean_metrics.errors, 0.95),
    cv_metrics.mean_score(),
    cv_metrics.rmse(), percentile(cv_metrics.errors, 0.95), ca_metrics.errors.size(),
    ca_metrics.mean_score(), ca_metrics.rmse(), percentile(ca_metrics.errors, 0.95),
    fused_metrics.mean_score(), fused_metrics.rmse(), percentile(fused_metrics.errors, 0.95),
    fused_euclidean_metrics.rmse(), percentile(fused_euclidean_metrics.errors, 0.95),
    baseline_gain);

  fmt::print(
    "\nPer-horizon metrics\n"
    "horizon_s,predictions,matched,coverage_pct,tracking_output_coverage_pct,hold_score_pct,"
    "hold_rmse_m,cv_score_pct,cv_rmse_m,ca_score_pct,ca_rmse_m,fused_score_pct,"
    "fused_rmse_m,fused_p95_m,fused_vs_hold_gain_pct,hit_at_0.1m_pct\n");
  double max_qualified_horizon = nan;
  for (const auto & [value, metrics] : horizon_metrics) {
    const double horizon_coverage =
      metrics.prediction_count == 0
        ? nan
        : 100.0 * static_cast<double>(metrics.matched_count) / metrics.prediction_count;
    const double horizon_tracking_coverage =
      processed_frames == 0
        ? nan
        : 100.0 * static_cast<double>(metrics.prediction_count) / processed_frames;
    const double horizon_hit =
      metrics.matched_count == 0
        ? nan
        : 100.0 * static_cast<double>(metrics.hit_at_01_count) / metrics.matched_count;
    const double horizon_baseline_gain =
      metrics.fused.mean_score() - metrics.hold.mean_score();
    fmt::print(
      "{:.3f},{},{},{:.3f},{:.3f},{:.3f},{:.6f},{:.3f},{:.6f},{:.3f},{:.6f},{:.3f},"
      "{:.6f},{:.6f},{:.3f},{:.3f}\n",
      value, metrics.prediction_count, metrics.matched_count, horizon_coverage,
      horizon_tracking_coverage,
      metrics.hold.mean_score(), metrics.hold.rmse(),
      metrics.cv.mean_score(), metrics.cv.rmse(), metrics.ca.mean_score(), metrics.ca.rmse(),
      metrics.fused.mean_score(), metrics.fused.rmse(), percentile(metrics.fused.errors, 0.95),
      horizon_baseline_gain, horizon_hit);
    if (
      !metrics.fused.errors.empty() && metrics.fused.mean_score() >= min_score &&
      metrics.matched_count >= static_cast<std::size_t>(min_matched) &&
      metrics.hold.errors.size() == metrics.fused.errors.size() &&
      horizon_baseline_gain >= min_baseline_gain && horizon_coverage >= min_coverage &&
      horizon_tracking_coverage >= min_tracking_coverage && input_coverage >= min_input_coverage &&
      !timestamp_integrity_error) {
      max_qualified_horizon = value;
    }
  }
  const bool target_met = std::isfinite(max_qualified_horizon);
  fmt::print(
    "required_score_pct={:.3f}\nrequired_coverage_pct={:.3f}\n"
    "required_tracking_output_coverage_pct={:.3f}\nrequired_matched_count={}\n"
    "required_input_coverage_pct={:.3f}\n"
    "required_baseline_gain_pct={:.3f}\n"
    "max_qualified_horizon_s={:.6f}\ntarget_met={}\n",
    min_score, min_coverage, min_tracking_coverage, min_matched, min_input_coverage,
    min_baseline_gain, max_qualified_horizon, target_met);

  const std::string metadata_path = output_path + ".meta.json";
  nlohmann::json metadata{
    {"input_path", input_path},
    {"truth_source", "robust_zero_phase_raw_pnp_trajectory"},
    {"truth_is_external_ground_truth", false},
    {"truth_smoothing_window_s", truth_smoothing_window},
    {"score_mode", score_mode},
    {"score_ignores_radial_error", score_mode == "tracking"},
    {"score_sigma_m", score_sigma},
    {"horizons_s", horizons},
    {"processed_frames", processed_frames},
    {"expected_input_frames", expected_input_frames},
    {"input_coverage_pct", input_coverage},
    {"prediction_count", predictions.size()},
    {"matched_count", matched_count},
    {"match_coverage_pct", coverage},
    {"tracking_output_coverage_pct", tracking_output_coverage},
    {"fused_score_pct", fused_metrics.mean_score()},
    {"fused_scored_rmse_m", fused_metrics.rmse()},
    {"fused_scored_p95_m", percentile(fused_metrics.errors, 0.95)},
    {"fused_euclidean_rmse_m", fused_euclidean_metrics.rmse()},
    {"fused_euclidean_p95_m", percentile(fused_euclidean_metrics.errors, 0.95)},
    {"hold_score_pct", hold_metrics.mean_score()},
    {"fused_vs_hold_gain_pct", baseline_gain},
    {"required_score_pct", min_score},
    {"required_match_coverage_pct", min_coverage},
    {"required_tracking_output_coverage_pct", min_tracking_coverage},
    {"required_input_coverage_pct", min_input_coverage},
    {"required_matched_count", min_matched},
    {"required_baseline_gain_pct", min_baseline_gain},
    {"max_qualified_horizon_s", max_qualified_horizon},
    {"target_met", target_met}};
  std::ofstream metadata_output(metadata_path);
  if (!metadata_output) {
    fmt::print(stderr, "Failed to open metadata output: {}\n", metadata_path);
    return 1;
  }
  metadata_output << std::setw(2) << metadata << '\n';
  fmt::print("metadata_output={}\n", metadata_path);

  if (matched_count == 0) {
    fmt::print(stderr, "No prediction had an independent future observation bracket.\n");
    return 3;
  }
  if (timestamp_integrity_error) return 5;
  if (!target_met) return 4;
  return 0;
}
