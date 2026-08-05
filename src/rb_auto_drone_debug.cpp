#include <fmt/core.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>  
#include <opencv2/opencv.hpp>
#include <thread>

// 底层 IO 与工具
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp" 
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/prediction_cadence.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

// 无人机自瞄算法模块
#include "tasks/auto_drone/drone_yolo.hpp"
#include "tasks/auto_drone/drone_solver.hpp"
#include "tasks/auto_drone/drone_tracker.hpp"
#include "tasks/auto_drone/drone_planner.hpp"

using namespace std::chrono_literals;

constexpr double AIM_OFFSET_STEP_DEG = 0.005;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{center-log     | build/diagnostics/center_distance_latest.csv | 中心误差连续采集CSV }"
  "{@config-path   | ../configs/auto_drone.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  // 1. 解析命令行与配置文件参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const std::filesystem::path center_log_path = cli.get<std::string>("center-log");
  if (!center_log_path.parent_path().empty()) {
    std::filesystem::create_directories(center_log_path.parent_path());
  }
  std::ofstream center_log(center_log_path);
  if (!center_log) {
    tools::logger()->error("Unable to open center-distance CSV: {}", center_log_path.string());
    return 2;
  }
  center_log << "time_s,frame_id,target_present,pnp_distance_m,track_distance_m,center_dx_px,"
                "center_dy_px,gimbal_yaw_deg,gimbal_pitch_deg,tracker_state,"
                "visual_yaw_correction_deg,visual_pitch_correction_deg\n";
  center_log << std::setprecision(10);
  const auto center_log_start = std::chrono::steady_clock::now();
  auto center_log_last_flush = center_log_start;
  tools::logger()->info("Center-distance CSV: {}", center_log_path.string());

  const auto config = tools::load(config_path);
  const int prediction_frames_between_detections =
    config["prediction_frames_between_detections"]
      ? config["prediction_frames_between_detections"].as<int>()
      : 1;
  if (prediction_frames_between_detections < 0) {
    tools::logger()->error("prediction_frames_between_detections must be zero or positive");
    return 2;
  }

  // 2. 硬件 IO 初始化
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 3. 算法核心模块初始化
  auto_drone::YOLO yolo(config_path, true);
  auto_drone::Solver solver(config_path);
  auto_drone::Tracker tracker(config_path, &solver);
  tracker.set_gimbal(&gimbal); // 传入云台以获取敌方颜色状态
  auto_drone::Planner planner(config_path);
  // tools::Recorder record;

  // 4. 多线程通信队列 (容量设为1，保证规划线程总是拿到最新的目标)
  tools::ThreadSafeQueue<std::optional<auto_drone::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<double> current_fps(0.0);
  tools::PredictionCadence prediction_cadence(prediction_frames_between_detections);
  int return_code = 0;

  // =================================================================
  // 线程 A：云台规划控制与数据记录线程 (高频独立运行)
  // =================================================================
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;
    int plot_count = 0;
    auto last_plot_time = std::chrono::steady_clock::now();
    int current_freq = 0;
    auto next_control_time = std::chrono::steady_clock::now();

    while (!quit) {
      // 获取最新目标与云台状态
      auto target = target_queue.front(); 
      auto gs = gimbal.state();

      float target_yaw = 0.0f, target_pitch = 0.0f, plan_yaw = 0.0f, plan_pitch = 0.0f;
      float target_x = 0.0f, target_y = 0.0f, target_z = 0.0f, target_distance = 0.0f;

      if (target.has_value()) {
        // MPC 弹道预测与控制解算
        auto plan = planner.plan(target, gs.bullet_speed);

        // 发送控制指令给下位机
        gimbal.drone_send(
          plan.control, plan.fire, 
          plan.yaw * 57.3, plan.yaw_vel, plan.yaw_acc, 
          plan.pitch * 57.3, plan.pitch_vel, plan.pitch_acc
        );
        
        

        auto fired = gs.bullet_count > last_bullet_count;
        last_bullet_count = gs.bullet_count;

        target_yaw = plan.target_yaw * 57.3f;
        target_pitch = plan.target_pitch * 57.3f;
        plan_yaw = plan.yaw * 57.3f;
        plan_pitch = plan.pitch * 57.3f;
        
        const auto xyz = target->get_xyz();
        target_x = xyz.x();
        target_y = xyz.y();
        target_z = xyz.z();
        target_distance = xyz.norm();
      } 
      else {
        // 丢失目标，向云台发送当前姿态的空闲指令（防暴走）
        gimbal.drone_send(
          false, false, gs.yaw * 57.3f, 0.0f, 0.0f, gs.pitch * 57.3f, 0.0f, 0.0f);
      }

      // --- 数据绘图与输出 (Plotter) ---
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_pitch"] = gs.pitch;
      data["target_yaw"] = target_yaw;
      data["target_pitch"] = target_pitch;
      data["plan_yaw"] = plan_yaw;
      data["plan_pitch"] = plan_pitch;
      data["target_x"] = target_x;
      data["target_y"] = target_y;
      data["target_z"] = target_z;
      data["target_distance"] = target_distance;
      data["fps"] = current_fps.load();

      plot_count++;
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - last_plot_time).count() >= 1) {
        current_freq = plot_count;
        plot_count = 0;
        last_plot_time = now;
      }
      data["send_freq"] = current_freq;

      plotter.plot(data);

      const auto control_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(prediction_cadence.control_period_s()));
      next_control_time += control_period;
      const auto after_control = std::chrono::steady_clock::now();
      if (next_control_time <= after_control) next_control_time = after_control + control_period;
      std::this_thread::sleep_until(next_control_time);
    }
  });

  // =================================================================
  // 主线程：图像获取、视觉解算与画面渲染 (跟随相机帧率)
  // =================================================================
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point last_capture_t;
  auto detection_window_start = std::chrono::steady_clock::now();
  std::uint64_t frame_id = 0;
  int detection_window_count = 0;
  double detection_fps = 0.0;

  while (!exiter.exit()) {
    camera.read(img, t);

    double capture_fps = current_fps.load();
    if (last_capture_t != std::chrono::steady_clock::time_point{} && t > last_capture_t) {
      capture_fps = 1.0 / std::chrono::duration<double>(t - last_capture_t).count();
      current_fps = capture_fps;
    }
    last_capture_t = t;

    std::optional<auto_drone::YOLOResult> result;
    try {
      result = yolo.detect_async(img, t, frame_id++);
    } catch (const std::exception & e) {
      tools::logger()->error("[AutoDrone] Inference failed, stopping control: {}", e.what());
      target_queue.push(std::nullopt);
      return_code = 1;
      break;
    }
    if (!result) continue;

    img = std::move(result->frame);
    t = result->timestamp;
    auto drones = std::move(result->drones);
    prediction_cadence.observe(t);

    detection_window_count++;
    const auto now = std::chrono::steady_clock::now();
    const double detection_window_s =
      std::chrono::duration<double>(now - detection_window_start).count();
    if (detection_window_s >= 1.0) {
      detection_fps = detection_window_count / detection_window_s;
      detection_window_count = 0;
      detection_window_start = now;
    }
    const double latency_ms = std::chrono::duration<double, std::milli>(now - t).count();

    // The pose must be queried with the timestamp of the completed inference frame.
    auto q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    // 解析当前云台的真实角度用于显示
    auto ypr = tools::eulers(q, 2, 1, 0);
    float yaw_deg = ypr[0] * 180.0 / M_PI;
    float pitch_deg = ypr[1] * 180.0 / M_PI;

    auto targets = tracker.track(drones, t);

    double raw_pnp_distance_m = std::numeric_limits<double>::quiet_NaN();
    double tracked_distance_m = std::numeric_limits<double>::quiet_NaN();
    cv::Point2f center_error_px(
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    if (!drones.empty()) {
      raw_pnp_distance_m = drones.front().xyz_in_gimbal.norm();
      center_error_px =
        drones.front().center - cv::Point2f(img.cols * 0.5F, img.rows * 0.5F);
    }
    if (!targets.empty()) tracked_distance_m = targets.front().get_xyz().norm();

    if (!targets.empty() && !drones.empty()) {
      planner.update_visual_feedback(center_error_px.x, center_error_px.y, t);
    } else {
      planner.reset_visual_feedback();
    }

    const auto center_log_now = std::chrono::steady_clock::now();
    center_log << std::chrono::duration<double>(center_log_now - center_log_start).count() << ','
               << result->frame_id << ',' << (!drones.empty()) << ',' << raw_pnp_distance_m << ','
               << tracked_distance_m << ',' << center_error_px.x << ',' << center_error_px.y << ','
               << yaw_deg << ',' << pitch_deg << ',' << tracker.state() << ','
               << planner.visual_yaw_correction_deg() << ','
               << planner.visual_pitch_correction_deg() << '\n';
    if (center_log_now - center_log_last_flush >= 1s) {
      center_log.flush();
      center_log_last_flush = center_log_now;
    }

    // 把目标塞给控制线程
    if (!targets.empty()) {
      target_queue.push(targets.front());
    } else {
      target_queue.push(std::nullopt);
    }

    // ---------------------- 画面渲染 (Debug 级别) ----------------------
    // 1. 打印基础信息
    tools::draw_text(img, fmt::format("Capture FPS: {:.1f}", capture_fps), {40, 40}, {0, 255, 0});
    tools::draw_text(img, fmt::format("Detection FPS: {:.1f}", detection_fps), {40, 80}, {0, 255, 0});
    tools::draw_text(img, fmt::format("Latency: {:.1f} ms", latency_ms), {40, 120}, {0, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "YOLO: {:.1f}/{:.1f}/{:.1f} ms, {} TensorRT streams, {} dropped",
        result->preprocess_ms, result->request_ms, result->postprocess_ms,
        yolo.inference_streams(), yolo.dropped_frames()),
      {40, 160}, {255, 255, 0});
    tools::draw_text(img, fmt::format("Gimbal Yaw: {:.2f}", yaw_deg), {40, 200}, {0, 128, 255});
    tools::draw_text(img, fmt::format("Gimbal Pitch: {:.2f}", pitch_deg), {40, 240}, {0, 255, 255});
    tools::draw_text(img, fmt::format("Tracker State: {}", tracker.state()), {40, 280}, {255, 255, 0});
    tools::draw_text(
      img, fmt::format("Aim Offset Yaw: {:+.3f} deg", planner.yaw_offset_deg()), {40, 320},
      {255, 128, 0});
    tools::draw_text(
      img, fmt::format("Aim Offset Pitch: {:+.3f} deg", planner.pitch_offset_deg()), {40, 360},
      {255, 128, 0});
    tools::draw_text(
      img, "W/S: Pitch +/-  A/D: Yaw -/+  (0.005 deg)", {40, 400}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "PnP Dist: {:.2f} m  Track Dist: {:.2f} m", raw_pnp_distance_m, tracked_distance_m),
      {40, 440}, {80, 230, 255});
    tools::draw_text(
      img, fmt::format("Center Error: dx={:+.1f}px dy={:+.1f}px", center_error_px.x,
                       center_error_px.y),
      {40, 480}, {80, 230, 255});
    tools::draw_text(
      img,
      fmt::format(
        "Visual Correction: yaw={:+.3f} pitch={:+.3f} deg",
        planner.visual_yaw_correction_deg(), planner.visual_pitch_correction_deg()),
      {40, 520}, {80, 230, 255});

    // 2. 绘制 YOLO 检测到的无人机 2D Bbox 和 关键点
    for (const auto& drone : drones) {
      cv::rectangle(img, drone.box, cv::Scalar(200, 255, 0), 2);
      // 将 kpts 改为 points
      for (const auto& pt : drone.points) {
        cv::circle(img, pt, 4, cv::Scalar(0, 255, 0), -1); 
      }
      cv::putText(img, fmt::format("{:.2f}", drone.confidence), drone.box.tl(), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    cv::circle(img,cv::Point2f(img.cols/2,img.rows/2),5,cv::Scalar(0,255,0),-1);

    // 缩小一半显示防止撑爆屏幕
    // record.record(img,q,t);
    
    cv::resize(img, img, {}, 0.5, 0.5);  
    cv::imshow("Auto Drone System", img);

    // 键盘事件处理
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    if (key == 'w' || key == 'W') planner.adjust_aim_offset(0.0, AIM_OFFSET_STEP_DEG);
    if (key == 's' || key == 'S') planner.adjust_aim_offset(0.0, -AIM_OFFSET_STEP_DEG);
    if (key == 'a' || key == 'A') planner.adjust_aim_offset(AIM_OFFSET_STEP_DEG, 0.0);
    if (key == 'd' || key == 'D') planner.adjust_aim_offset(-AIM_OFFSET_STEP_DEG, 0.0);
    if (key == 'r') {
      io::GimbalState* g_demo = gimbal.set_state_();
      g_demo->mode = !g_demo->mode;
    }
  }

  // =================================================================
  // 资源释放与安全退出
  // =================================================================
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  center_log.flush();
  
  // 发送归中或停止指令，防止下位机继续飞转
  auto current_state = gimbal.state();
  gimbal.drone_send(
      false, 
      false, 
      current_state.yaw * 57.3f,
      0.0f, 
      0.0f, 
      current_state.pitch * 57.3f,
      0.0f, 
      0.0f
  );

  return return_code;
}
