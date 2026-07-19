#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
// #include "tasks/auto_aim/detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | ../configs/xiaohei.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{headless       | false             | 无窗口健康检查模式 }"
  "{@input-path    | ../快/2026-04-04_19-58-26 | avi和txt文件的路径}";

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
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto headless = cli.get<bool>("headless");
  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);
  if (!video.isOpened() || !text.is_open()) {
    tools::logger()->error("Cannot open demo input {}(.avi/.txt)", input_path);
    return 2;
  }

  auto_aim::YOLO yolo(config_path);
  // auto_aim::Detector traditional(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, &solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img, drawing;
  auto t0 = std::chrono::steady_clock::now();

  auto_aim::Target last_target;
  io::Command last_command;
  bool tracking_seen = false;
  bool health_ok = true;
  int processed_frames = 0;
  int eligible_update_frames = 0;
  int matched_update_frames = 0;

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;
    // auto inshow_start = std::chrono::steady_clock::now();
    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    if (!(text >> t >> w >> x >> y >> z)) {
      tools::logger()->error("Demo pose stream ended before video at frame {}", frame_count);
      health_ok = false;
      break;
    }
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));
    ++processed_frames;

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    // auto traditional_start = std::chrono::steady_clock::now();
    // auto armors = traditional.detect(img, frame_count);

    const std::string tracker_state_before = tracker.state();
    const bool has_expected_detection = last_target.checkinit() &&
      std::any_of(armors.begin(), armors.end(), [&](const auto_aim::Armor & armor) {
        return armor.name == last_target.name && armor.type == last_target.armor_type;
      });
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.test_track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    if (!headless) {
      tools::logger()->info(
        "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
        tools::delta_time(tracker_start, yolo_start) * 1e3,
        tools::delta_time(aimer_start, tracker_start) * 1e3,
        tools::delta_time(finish, aimer_start) * 1e3);
    }

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
      if (armor.pnp_valid) {
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
        data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      }
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    Eigen::Quaternion q{w, x, y, z};
    auto yaw = tools::eulers(q, 2, 1, 0)[0];
    data["gimbal_yaw"] = yaw * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    const bool eligible_candidate = tracker_state_before != "lost" && has_expected_detection;

    if (!targets.empty()) {
      auto target = targets.front();
      const bool eligible_update = eligible_candidate && target.update_count_ > 0;
      if (eligible_update) ++eligible_update_frames;
      tracking_seen = tracking_seen || tracker.state() == "tracking";

      const bool finite_target = target.center().allFinite() && target.velocity().allFinite() &&
        target.rotation().allFinite() && std::isfinite(target.yaw()) &&
        std::isfinite(target.yaw_rate()) && target.covariance().allFinite();
      if (!finite_target || target.diverged()) health_ok = false;
      for (int id = 0; id < target.armor_count(); ++id) {
        const double radius = target.radius(id);
        if (!std::isfinite(radius) ||
            (target.name != auto_aim::ArmorName::base &&
             (radius < auto_aim::motion_model::MIN_RADIUS ||
              radius > auto_aim::motion_model::MAX_RADIUS))) {
          health_ok = false;
        }
      }
      for (const auto & pose : target.armor_pose_list()) {
        if (!pose.matrix().allFinite()) health_ok = false;
      }

      const auto diagnostics = target.estimator_diagnostics();
      if (
        !std::isfinite(diagnostics.residual_norm) || !std::isfinite(diagnostics.nis) ||
        !std::isfinite(diagnostics.normalized_nis)) {
        health_ok = false;
      }
      if (eligible_update) {
        if (diagnostics.observation_dim > 0) ++matched_update_frames;
        else health_ok = false;
      }

      // 当前帧target更新后
      for (const auto & pose : target.armor_pose_list()) {
        auto image_points = solver.reproject_pose(pose, target.armor_type);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

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
      data["r"] = target.radius(0);
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      data["residual_norm"] = diagnostics.residual_norm;
      data["nis"] = diagnostics.nis;
      data["observation_dim"] = diagnostics.observation_dim;
      data["normalized_nis"] = diagnostics.normalized_nis;
      last_target = target;
    } else if (eligible_candidate && last_target.update_count_ > 0) {
      ++eligible_update_frames;
      health_ok = false;
    }

    if (!headless) {
      plotter.plot(data);
      cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
      cv::imshow("reprojection", img);
       int key = cv::waitKey(100);
    if (key == 'q') break;
    while (key == ' ') {
      int y = cv::waitKey(30);
      if (y == 'q') break;
    }
    }

    //  tools::logger()->info(
    //     "imshow : {:.1f}ms",  tools::delta_time(std::chrono::steady_clock::now(), inshow_start) * 1e3);
  }

  if (headless) {
    if (processed_frames == 0 || !tracking_seen || eligible_update_frames == 0 ||
        matched_update_frames != eligible_update_frames) {
      health_ok = false;
    }
    tools::logger()->info(
      "Headless replay: frames={}, tracking={}, matched_updates={}/{}", processed_frames,
      tracking_seen, matched_update_frames, eligible_update_frames);
    if (!health_ok) return 3;
  }

  return 0;
}
