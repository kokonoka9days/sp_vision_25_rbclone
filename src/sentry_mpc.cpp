#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"  // 改为使用Gimbal串口通信
#include "io/ros2/publish2nav.hpp"
#include "io/ros2/ros2.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/planner/planner.hpp"  // MPC 规划器
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry_mpc.yaml | 位置参数，yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  // ROS2 通信
  io::ROS2 ros2;
  
  // 主相机（工业相机）
  io::Camera camera(config_path);
  
  // 全向感知相机（工业相机）
  std::string omnl_yaml_name = "configs/omniperception/omn_camera_left.yaml";
  std::string omnr_yaml_name = "configs/omniperception/omn_camera_right.yaml";
  io::Camera omn_cam1(omnl_yaml_name);
  io::Camera omn_cam2(omnr_yaml_name);
  auto omn_l_yaml = tools::load(omnl_yaml_name);
  auto omn_r_yaml = tools::load(omnr_yaml_name);
  omn_cam1.main_and_secondary = tools::read<std::string>(omn_l_yaml, "main_and_secondary");
  omn_cam2.main_and_secondary = tools::read<std::string>(omn_r_yaml, "main_and_secondary");
  // io::Camera back_camera("configs/camera.yaml");
  
  // 改为使用Gimbal串口通信（替代CBoard）
  io::Gimbal gimbal(config_path);
  
  // 视觉模块
  auto_aim::YOLO yolo(config_path, false);  // 主相机YOLO
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  
  
  // MPC 规划器
  auto_aim::Planner planner(config_path);
  
  // 全向感知决策器
  omniperception::Decider decider(config_path);
  
  // 线程安全队列（用于MPC规划线程）
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);
  
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  
  // 云台状态
  Eigen::Vector3d gimbal_euler;
  
  // 获取云台模式
  auto last_mode = io::GimbalMode::IDLE;
  
  // MPC 规划线程（独立线程运行MPC控制器）
  std::atomic<bool> quit = false;
  auto mpc_thread = std::thread([&]() {
    auto_aim::Plan current_plan{false};
    
    while (!quit) {
      if (!target_queue.empty()) {
        auto target = target_queue.front();
        
        if (target.has_value()) {
          // 使用MPC规划器计算控制指令
          auto gs = gimbal.state();
          current_plan = planner.plan(*target, gs.bullet_speed);
          
          if (current_plan.control) {
            // 发送MPC控制指令到云台
            gimbal.send(
              current_plan.control,
              current_plan.fire,
              static_cast<float>(current_plan.yaw),
              static_cast<float>(current_plan.yaw_vel),
              static_cast<float>(current_plan.yaw_acc),
              static_cast<float>(current_plan.pitch),
              static_cast<float>(current_plan.pitch_vel),
              static_cast<float>(current_plan.pitch_acc)
            );
          }
        } 
      }
      std::this_thread::sleep_for(10ms);
    }
  });

  // 主循环
  while (!exiter.exit()) {
    // 读取云台模式
    auto mode = gimbal.mode();
    
    // 模式切换日志
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode;
    }
    
    // 只处理自瞄模式
    if (mode != io::GimbalMode::AUTO_AIM) {
      // 非自瞄模式：发送停止指令并跳过
      std::this_thread::sleep_for(50ms);
      continue;
    }
    
    // 读取主相机图像
    camera.read(img, timestamp);
    
    // 获取云台姿态（四元数）
    Eigen::Quaterniond q = gimbal.q(timestamp);
    
    // 更新解算器姿态
    solver.set_R_gimbal2world(q);
    
    // 获取云台欧拉角
    gimbal_euler = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    
    // 主相机检测
    auto armors = yolo.detect(img);
    
    // 更新无敌状态装甲板
    decider.get_invincible_armor(ros2.subscribe_enemy_status());
    
    // 过滤装甲板
    decider.armor_filter(armors);
    
    // 设置优先级
    decider.set_priority(armors);
    
    // 跟踪目标
    auto targets = tracker.track(armors, timestamp);
    
    // 模式判断：如果跟踪器丢失目标，切换到全向感知模式
    if (tracker.state() == "lost") {
      // 全向感知模式
      io::sb_VisionToGimbal vision_cmd = decider.decide_g(
        yolo, gimbal_euler, omn_cam1, omn_cam2);
      
      if (vision_cmd.mode != 0) {

        vision_cmd.work_mode = static_cast<uint8_t>(io::WorkMode::OMNI_PERCEPTION);
        // 全向感知找到目标，发送控制指令
        // 使用Gimbal的send函数直接发送VisionToGimbal结构体
        gimbal.sb_send(vision_cmd);
        
        // 射击判断（使用shooter_g版本）
        bool should_shoot = shooter.shoot_g(vision_cmd, aimer, targets, gimbal_euler);
        
        // 如果需要射击且vision_cmd.mode是2（控制开火），则发送射击指令
        if (should_shoot && vision_cmd.mode == 2) {
          // 重新发送带有射击标志的指令
          vision_cmd.mode = 2;  // 控制并开火
          gimbal.sb_send(vision_cmd);
        }
        else{
          gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        }
      }
    } else {
      // 自瞄模式 - 使用MPC
      if (!targets.empty()) {
        // 将目标放入队列供MPC线程处理
        target_queue.push(targets.front());
      } else {
        target_queue.push(std::nullopt);
      }
    }
    
    // ROS2通信 - 发布目标信息
    Eigen::Vector4d target_info = decider.get_target_info(armors, targets);
    ros2.publish(target_info);
  }
  
  // 清理
  quit = true;
  if (mpc_thread.joinable()) {
    mpc_thread.join();
  }
  
  // 发送停止指令
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  
  return 0;
}