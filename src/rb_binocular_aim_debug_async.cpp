#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "method_set/binocular_aim.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{short_camera   | ../configs/sb_short.yaml | 短焦相机配置文件路径}"
  "{long_camera    | ../configs/sb_long.yaml  | 长焦相机配置文件路径}";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto short_camera_config_path = cli.get<std::string>("short_camera");
  const auto long_camera_config_path = cli.get<std::string>("long_camera");
  if (short_camera_config_path.empty() || long_camera_config_path.empty()) {
    cli.printMessage();
    return 1;
  }

  io::Camera::initSDK();
  io::Camera short_camera(short_camera_config_path);
  io::Camera long_camera(long_camera_config_path);
  io::Gimbal gimbal(short_camera_config_path);

  auto_aim::YOLO yolo(short_camera_config_path, false);
  auto_aim::Solver short_camera_solver(short_camera_config_path);
  auto_aim::Solver long_camera_solver(long_camera_config_path);
  auto_aim::Tracker tracker(short_camera_config_path, &short_camera_solver);
  tracker.set_gimbal(&gimbal);
  auto_aim::Planner short_camera_planner(short_camera_config_path);
  auto_aim::Planner long_camera_planner(long_camera_config_path);

  BinocularAim binocular_aim(
    short_camera, long_camera, short_camera_solver, long_camera_solver, short_camera_planner,
    long_camera_planner);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::mutex short_planner_mutex;
  std::mutex long_planner_mutex;
  std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();

      const bool use_short_planner = !target.has_value() || target->cam_is_short;
      auto & planner = use_short_planner ? short_camera_planner : long_camera_planner;
      auto & planner_mutex = use_short_planner ? short_planner_mutex : long_planner_mutex;

      uint8_t name = 0;
      float tx = 0.0f;
      float ty = 0.0f;

      if (target.has_value()) {
          name = static_cast<uint8_t>(target->name) + 1;
          tx = target->ekf_x()[0]; 
          ty = target->ekf_x()[2]; 

          // tools::logger()->info("{},{},{}", name,tx,ty);

        }

      auto_aim::Plan plan{false};
      {
        std::lock_guard<std::mutex> lock(planner_mutex);
        plan = planner.plan(
          target, gs.bullet_speed, gs.yaw,
          auto_aim::Planner::ShootStrategy::rbSuppressiveFire);
      }

    //    if (binocular_aim.force_control_frames > 0) {
    //     plan.control = true;
    //     binocular_aim.force_control_frames--;
    // }

     gimbal.sb_send(
        plan.control, plan.fire,
        plan.yaw, plan.yaw_vel, plan.yaw_acc,
        plan.pitch, plan.pitch_vel, plan.pitch_acc,
        tx,ty,name
      );    
      const auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;
      data["q2yaw"] = gs.q2yaw;
      data["q2pitch"] = gs.q2pitch;
      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;
      data["plan_mode"] = plan.control ? (plan.fire ? 2 : 1) : 0;
      data["plan_yaw"] = plan.yaw / CV_PI * 180.;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;
      data["plan_pitch"] = plan.pitch * 57.3;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;
      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;
      data["camera_is_short"] = use_short_planner ? 1 : 0;

      if (target.has_value()) {
        const auto ekf = target->ekf_x();
        data["target_z"] = ekf[4];
        data["target_vz"] = ekf[5];
        data["tower_h1"] = target->tower_armor_hs[0];
        data["tower_h2"] = target->tower_armor_hs[1];
        data["tower_h3"] = target->tower_armor_hs[2];
        data["tower_armor_h"] = target->tower_armor_h;
        data["ekf_x"] = ekf(0);
        data["ekf_vx"] = ekf(1);
        data["ekf_y"] = ekf(2);
        data["ekf_vy"] = ekf(3);
        data["ekf_z"] = ekf(4);
        data["ekf_vz"] = ekf(5);
        data["ekf_yaw"] = ekf(6) * 57.3;
        data["ekf_vyaw"] = ekf(7) * 57.3;
        data["ekf_r"] = ekf(8);
      }

      plotter.plot(data);
      std::this_thread::sleep_for(10ms);
    }
  });

  struct PendingFrame
  {
    std::chrono::steady_clock::time_point timestamp;
    bool is_short;
    std::uint64_t generation;
  };

  struct LongCameraHandover
  {
    bool active = false;
    std::uint64_t generation = 0;
    std::chrono::steady_clock::time_point started_at;
    int consecutive_misses = 0;
    auto_aim::ArmorName target_name = auto_aim::ArmorName::not_armor;
    auto_aim::ArmorType target_type = auto_aim::ArmorType::small;
  };

  constexpr auto long_handover_timeout = 150ms;
  constexpr int long_handover_max_misses = 3;

  std::deque<PendingFrame> pending_frames;
  LongCameraHandover long_handover;
  std::optional<std::chrono::steady_clock::time_point> last_tracker_timestamp;
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point last_t;
  int frame_count = 0;

  while (!exiter.exit()) {
    binocular_aim.cameras.aim_ptr->read(img, t);
    if (img.empty()) continue;

    const bool input_is_short = binocular_aim.is_short;
    const auto input_generation = binocular_aim.generation();
    const auto timestamp_offset =
      input_is_short ? short_camera.timestamp_offset : long_camera.timestamp_offset;
    auto q = gimbal.q(t);
    if (last_t != std::chrono::steady_clock::time_point{}) {
      const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t - last_t).count();
      if (elapsed_us > 0) {
        tools::logger()->info("capture fps: {:.2f}", 1000000. / elapsed_us);
      }
    }
    last_t = t;

    pending_frames.push_back({t, input_is_short, input_generation});
    auto yolo_frame = yolo.detect(auto_aim::YOLOFrameData(img, q, t), frame_count++);
    if (yolo_frame.is_empty) {
      continue;
    }

    const auto pending_frame = std::find_if(
      pending_frames.begin(), pending_frames.end(), [&](const PendingFrame & frame) {
        return frame.timestamp == yolo_frame.timestamp;
      });
    if (pending_frame == pending_frames.end()) {
      tools::logger()->warn("[BinocularAim] 无法确定异步检测结果的相机来源");
      continue;
    }

    const bool frame_is_short = pending_frame->is_short;
    const auto frame_generation = pending_frame->generation;
    pending_frames.erase(pending_frame);

    img = yolo_frame.frame;
    q = yolo_frame.gimbal_q;
    t = yolo_frame.timestamp;

    if (
      frame_generation != binocular_aim.generation() ||
      frame_is_short != binocular_aim.is_short) {
      tools::logger()->debug(
        "[BinocularAim] 丢弃切换前的异步结果: frame_generation={}, current_generation={}",
        frame_generation, binocular_aim.generation());
      continue;
    }

    if (last_tracker_timestamp.has_value() && t <= *last_tracker_timestamp) {
      tools::logger()->warn("[BinocularAim] 丢弃时间戳未递增的检测结果");
      continue;
    }
    last_tracker_timestamp = t;

    auto & frame_solver = frame_is_short ? short_camera_solver : long_camera_solver;
    auto & frame_planner = frame_is_short ? short_camera_planner : long_camera_planner;
    auto & frame_planner_mutex = frame_is_short ? short_planner_mutex : long_planner_mutex;

    frame_solver.set_R_gimbal2world(q);
    tracker.setSolver(&frame_solver);

    auto armors = std::move(yolo_frame.armors);
    auto targets = tracker.track(armors, t, frame_is_short);

    if (long_handover.active) {
      if (
        frame_generation != long_handover.generation || frame_is_short ||
        binocular_aim.is_short) {
        long_handover.active = false;
      } else {
        const bool target_detected = std::any_of(
          armors.begin(), armors.end(), [&](const auto_aim::Armor & armor) {
            return armor.name == long_handover.target_name &&
                   armor.type == long_handover.target_type;
          });

        if (target_detected) {
          tools::logger()->info("[BinocularAim] 长焦已接管目标");
          long_handover.active = false;
        } else {
          long_handover.consecutive_misses++;
          const auto handover_elapsed = std::chrono::steady_clock::now() - long_handover.started_at;
          if (
            handover_elapsed >= long_handover_timeout &&
            long_handover.consecutive_misses >= long_handover_max_misses &&
            binocular_aim.Switch(tracker, true)) {
            tools::logger()->warn(
              "[BinocularAim] 长焦接管失败，连续 {} 帧未检测到原目标，回退短焦",
              long_handover.consecutive_misses);
            long_handover.active = false;
            target_queue.push(std::nullopt);
            continue;
          }
        }
      }
    }

    const auto ypr = tools::eulers(q, 2, 1, 0);
    const float yaw_deg = ypr[0] * 180.0 / M_PI;
    const float pitch_deg = ypr[1] * 180.0 / M_PI;
    tools::draw_text(img, fmt::format("rb_Yaw {:.2f}", yaw_deg), {40, 40}, {0, 128, 255});
    tools::draw_text(
      img, fmt::format("rb_Pitch {:.2f}", pitch_deg), {40, 80}, {0, 255, 255});
    tools::draw_text(
      img, frame_is_short ? "Camera: short" : "Camera: long", {40, 120}, {255, 255, 0});

    if (targets.empty()) {
      target_queue.push(std::nullopt);
    } else {
      target_queue.push(targets.front());
      auto & target = targets.front();
      const auto ekf_x = target.getEKFXest();

      const Eigen::Vector3d center_world(ekf_x[0], ekf_x[2], ekf_x[4]);
      const Eigen::Vector3d velocity(ekf_x[1], ekf_x[3], ekf_x[5]);
      const Eigen::Vector3d pred_center = center_world + velocity * 0.5;
      Eigen::Vector3d v_yaw_axis_tvec = center_world;
      v_yaw_axis_tvec[2] += ekf_x[7] * 0.1;

      const auto center_img =
        frame_solver.reproject_armor(center_world, 0.0, target.armor_type, target.name);
      const auto pred_point_img =
        frame_solver.reproject_armor(pred_center, 0.0, target.armor_type, target.name);
      const auto v_yaw_axis_point_img =
        frame_solver.reproject_armor(v_yaw_axis_tvec, 0.0, target.armor_type, target.name);

      if (!center_img.empty() && !pred_point_img.empty()) {
        cv::circle(img, center_img[0], 5, cv::Scalar(51, 153, 237), -1);
        cv::circle(img, pred_point_img[0], 8, cv::Scalar(0, 0, 255), -1);
        cv::line(img, center_img[0], pred_point_img[0], cv::Scalar(0, 255, 255), 2);
        if (!v_yaw_axis_point_img.empty()) {
          cv::line(img, center_img[0], v_yaw_axis_point_img[0], cv::Scalar(0, 255, 0), 2);
        }
      }

      for (const auto & xyza : target.armor_xyza_list()) {
        const auto image_points =
          frame_solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {235, 206, 135});
      }

      Eigen::Vector4d aim_xyza;
      {
        std::lock_guard<std::mutex> lock(frame_planner_mutex);
        aim_xyza = frame_planner.debug_xyza;
      }
      const auto aim_points = frame_solver.reproject_armor(
        aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, aim_points, {0, 0, 255});

      if (binocular_aim.ChangeTheScope(target, tracker) && !binocular_aim.is_short) {
        long_handover.active = true;
        long_handover.generation = binocular_aim.generation();
        long_handover.started_at = std::chrono::steady_clock::now();
        long_handover.consecutive_misses = 0;
        long_handover.target_name = target.name;
        long_handover.target_type = target.armor_type;
        tools::logger()->info(
          "[BinocularAim] 开始长焦接管，generation={}", long_handover.generation);
      }
    }

  //   cv::resize(img, img, {}, 0.5, 0.5);
  //   cv::imshow("reprojection", img);
  //   const auto key = cv::waitKey(1);
  //   if (key == 'q') break;
  //   if (key == 'c') binocular_aim.Switch(tracker, true);
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();

  const auto current_state = gimbal.state();
  gimbal.send(
    false, false, current_state.yaw / 57.3f, 0.0f, 0.0f, current_state.pitch / 57.3f, 0.0f,
    0.0f);

  return 0;
}
