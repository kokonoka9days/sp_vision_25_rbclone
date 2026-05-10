#include <fmt/core.h>
#include <atomic>
#include <chrono>
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
#include "tools/thread_safe_queue.hpp"

// 无人机自瞄算法模块
#include "tasks/auto_drone/drone_yolo.hpp"
#include "tasks/auto_drone/drone_solver.hpp"
#include "tasks/auto_drone/drone_tracker.hpp"
#include "tasks/auto_drone/drone_planner.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
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

  // 2. 硬件 IO 初始化
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 3. 算法核心模块初始化
  auto_drone::YOLO yolo(config_path, true);
  auto_drone::Solver solver(config_path);
  auto_drone::Tracker tracker(config_path, &solver);
  tracker.set_gimbal(&gimbal); // 传入云台以获取敌方颜色状态
  auto_drone::Planner planner(config_path);

  // 4. 多线程通信队列 (容量设为1，保证规划线程总是拿到最新的目标)
  tools::ThreadSafeQueue<std::optional<auto_drone::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<double> current_fps(0.0);

  // =================================================================
  // 线程 A：云台规划控制与数据记录线程 (高频独立运行)
  // =================================================================
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;
    int plot_count = 0;
    auto last_plot_time = std::chrono::steady_clock::now();
    int current_freq = 0;

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
        gimbal.drone_send(false, false, gs.yaw, 0.0f, 0.0f, gs.pitch, 0.0f, 0.0f);
      }

      // --- 数据绘图与输出 (Plotter) ---
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_roll"] = gs.roll;
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

      // 控制频率：~200Hz
      std::this_thread::sleep_for(7ms);
    }
  });

  // =================================================================
  // 主线程：图像获取、视觉解算与画面渲染 (跟随相机帧率)
  // =================================================================
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point last_t = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    camera.read(img, t);

    // 获取插值后的欧拉角并传入 Solver (考虑相机与通信的延迟 3ms)
    auto ypr = gimbal.ypr(t);
    solver.set_R_gimbal2world(ypr[0], ypr[1], ypr[2]);

    // 帧率计算
    double fps = 1.0 / std::chrono::duration_cast<std::chrono::microseconds>(t - last_t).count() * 1000000;
    last_t = t;
    current_fps = fps;

    // 解析当前云台的真实角度用于显示
    float yaw_deg = ypr[0] * 57.3f;
    float pitch_deg = ypr[1] * 57.3f;

    // 核心视觉与追踪管线
    auto drones = yolo.detect(img);
    auto targets = tracker.track(drones, t);

    // 把目标塞给控制线程
    if (!targets.empty()) {
      target_queue.push(targets.front());
    } else {
      target_queue.push(std::nullopt);
    }

    // ---------------------- 画面渲染 (Debug 级别) ----------------------
    // 1. 打印基础信息
    tools::draw_text(img, fmt::format("FPS: {:.1f}", fps), {40, 40}, {0, 255, 0});
    tools::draw_text(img, fmt::format("Gimbal Yaw: {:.2f}", yaw_deg), {40, 80}, {0, 128, 255});
    tools::draw_text(img, fmt::format("Gimbal Pitch: {:.2f}", pitch_deg), {40, 120}, {0, 255, 255});
    tools::draw_text(img, fmt::format("Tracker State: {}", tracker.state()), {40, 160}, {255, 255, 0});

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

    // 3. 绘制 3D 重投影与预测落点
    if (!targets.empty()) {
      auto target = targets.front();
      
      // 获取 Planner 记录的真实空间瞄准坐标
      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      Eigen::Vector3d aim_xyz = aim_xyza.head(3);

      // (A) 画出预测瞄准的 3D 边框（利用 Solver 的八点重投影）
      // Eigen::Vector3d zero_ypr = {aim_xyza[3], 0.0, 0.0}; // 假设预测姿态只考虑 Yaw
      // auto reproj_points = solver.reproject_drone(aim_xyz, zero_ypr);
      // tools::draw_points(img, reproj_points, {0, 0, 255}); // 红色表示预测位置框

      // (B) 算出瞄准的中心点并画一条连接真实中心和预测中心的射击引导线
      // auto pred_center_img = solver.world2pixel({cv::Point3f(aim_xyz.x(), aim_xyz.y(), aim_xyz.z())});
      // auto real_center_img = solver.world2pixel({cv::Point3f(target.get_xyz().x(), target.get_xyz().y(), target.get_xyz().z())});
      
      // if (!pred_center_img.empty() && !real_center_img.empty()) {
      //   // 画出真实位置中心
      //   cv::circle(img, real_center_img[0], 6, cv::Scalar(51, 153, 237), -1);
      //   // 画出预测位置中心
      //   cv::circle(img, pred_center_img[0], 8, cv::Scalar(0, 0, 255), -1);
      //   // 画出提前量引导线
      //   cv::line(img, real_center_img[0], pred_center_img[0], cv::Scalar(0, 255, 255), 2);
      //   tools::draw_text(img, "AIM", cv::Point(pred_center_img[0].x + 10, pred_center_img[0].y), {0, 0, 255});
      // }
    }

    // 缩小一半显示防止撑爆屏幕
    cv::resize(img, img, {}, 0.5, 0.5);  
    cv::imshow("Auto Drone System", img);

    // 键盘事件处理
    auto key = cv::waitKey(1);
    if (key == 'q') break;
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
  
  // 发送归中或停止指令，防止下位机继续飞转
  auto current_state = gimbal.state();
  gimbal.drone_send(
      false, 
      false, 
      current_state.yaw / 57.3f, 
      0.0f, 
      0.0f, 
      current_state.pitch / 57.3f, 
      0.0f, 
      0.0f
  );

  return 0;
}