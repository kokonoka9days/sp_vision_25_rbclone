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
#include "tools/img_tools.hpp"

const std::string keys =
  "{help h usage ? |                                             | 输出命令行参数说明}"
  "{short_camera   | ../configs/sb.yaml                          | 短焦相机配置文件路径 }"
  "{long_camera    | ../configs/sb_copy.yaml                     | 长焦相机配置文件路径 }"
  "{l_cam          | ../configs/omniperception/short_camera.yaml | 左感知相机 }"
  "{r_cam          | ../configs/omniperception/long_camera.yaml  | 右感知相机 }";

using namespace std::chrono_literals;


template<typename T> 
class BinocularType{
public:
  using Ptr = T*;
  using Address = T&;

  Address short_aim, long_aim;
  Ptr aim_ptr = nullptr;
  BinocularType(T& short_aim_, T&long_aim_): short_aim(short_aim_), long_aim(long_aim_), aim_ptr(&short_aim_){}

  void Switch(){
    aim_ptr = aim_ptr == &short_aim ? & long_aim: &short_aim;
  }
};
struct BinocularAim{
  BinocularAim(
        io::Camera& cam_short, io::Camera& cam_long,
        auto_aim::Solver& solver_short, auto_aim::Solver& solver_long,
        auto_aim::Planner& planner_short, auto_aim::Planner& planner_long
    ) : cameras(cam_short, cam_long),
        solvers(solver_short, solver_long),
        planners(planner_short, planner_long) 
    {}
  BinocularType<io::Camera> cameras;
  BinocularType<auto_aim::Solver> solvers;
  BinocularType<auto_aim::Planner> planners;
  bool is_short = true;

  std::chrono::steady_clock::time_point switch_time_point;

  //长短焦各射程范围 min_near到max_far
  double short_min_near = 0, short_max_far = 3.3;
  double long_min_near = 1.5, long_max_far = 5.5;

  // 缓冲区 far2near and near2far
  double short2long_point =  3.4;//(short_max_far + long_min_near)/2.;
  double long2short_point = 2.5;

  /// @brief 长短焦强制切换
  void Switch(auto_aim::Tracker& tracker){
    this->cameras.Switch();
    this->solvers.Switch();
    this->planners.Switch();
    is_short = !is_short;
    switch_time_point = std::chrono::steady_clock::now();
    tracker.setSolver(this->solvers.aim_ptr);
  }

  /// @brief 长短焦自动切换逻辑
  void ChangeTheScope(auto_aim::Target target , auto_aim::Tracker& tracker){

    auto now = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - switch_time_point).count();
    
    if (elapsed_time < 1000) {
        return; // 在冷却时间内，直接退出，不执行切换判断
    }
    const auto x_est = target.getEKFXest();
    const double x = x_est(0), y = x_est(2), z = x_est(4);
    double dis = sqrt( x*x + y*y + z*z);

    // if(is_short ) dis -= 1.16;

    // tools::logger()->info("dis = {}", dis);
    
    if(is_short && dis > short2long_point ){
      tools::logger()->info("切换至长焦镜头, dis = {}", dis);
      Switch(tracker);
    }else if(!is_short && dis < long2short_point){
      tools::logger()->info("切换至短焦镜头 dis = {}", dis);
      Switch(tracker);
    }
  }
};


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
  auto short_camera_config_path = cli.get<std::string>("short_camera");
  auto long_camera_config_path = cli.get<std::string>("long_camera");

  // ROS2 通信
  io::ROS2 ros2;
  
  // 主相机（工业相机）
  io::Camera short_camera(short_camera_config_path);
  io::Camera long_camera(long_camera_config_path);
  
  
  // 全向感知相机（工业相机）
  std::string omnl_yaml_name = cli.get<std::string>("l_cam");
  std::string omnr_yaml_name = cli.get<std::string>("r_cam");
  io::Camera omn_cam1(omnl_yaml_name);
  io::Camera omn_cam2(omnr_yaml_name);
  auto omn_l_yaml = tools::load(omnl_yaml_name);
  auto omn_r_yaml = tools::load(omnr_yaml_name);
  omn_cam1.main_and_secondary = tools::read<std::string>(omn_l_yaml, "main_and_secondary");
  omn_cam2.main_and_secondary = tools::read<std::string>(omn_r_yaml, "main_and_secondary");
  // io::Camera back_camera("configs/camera.yaml");
  
  // 改为使用Gimbal串口通信（替代CBoard）
  io::Gimbal gimbal(short_camera_config_path);
  
  // 视觉模块
  auto_aim::YOLO yolo(short_camera_config_path, false);  // 主相机YOLO

  auto_aim::Solver short_camera_solver(short_camera_config_path);
  auto_aim::Solver long_camera_solver(long_camera_config_path);

  auto_aim::Tracker tracker(short_camera_config_path, &short_camera_solver);//默认短焦
  tracker.set_gimbal(&gimbal);
  // auto_aim::Aimer aimer(short_camera_config_path);
  // auto_aim::Shooter shooter(short_camera_config_path);
  
  // MPC 规划器
  auto_aim::Planner short_camera_planner(short_camera_config_path);
  auto_aim::Planner long_camera_planner(long_camera_config_path);

  //双目切换
  BinocularAim bincameras(short_camera, long_camera, 
                          short_camera_solver, long_camera_solver, 
                          short_camera_planner, long_camera_planner );
  
  // 全向感知决策器
  omniperception::Decider decider(omnl_yaml_name);
  
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
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;
    
    while (!quit) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        
        // // 使用MPC规划器计算控制指令
        // plan = bincameras.planners.aim_ptr->plan(*target, 22);
          
        //MPC预测以及+自家火控
        auto_aim::Planner * plan_short_or_long = target->cam_is_short ? &bincameras.planners.short_aim : &bincameras.planners.long_aim;
        auto plan =  plan_short_or_long->plan(target, gs.bullet_speed, gs.yaw,  auto_aim::Planner::ShootStrategy::rbSuppressiveFire);
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        auto fired = gs.bullet_count > last_bullet_count;
        last_bullet_count = gs.bullet_count;
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

        data["gimbal_yaw"] = gs.yaw;
        data["gimbal_pitch"] = gs.pitch;
        data["gimbal_pitch_vel"] = gs.pitch_vel;

        data["target_yaw"] = plan.target_yaw;
        data["target_pitch"] = plan.target_pitch;

        data["plan_yaw"] = plan.yaw;
        data["plan_yaw_vel"] = plan.yaw_vel;
        data["plan_yaw_acc"] = plan.yaw_acc;

        data["plan_pitch"] = plan.pitch;
        data["plan_pitch_vel"] = plan.pitch_vel;
        data["plan_pitch_acc"] = plan.pitch_acc;

        data["fire"] = plan.fire ? 1 : 0;
        data["fired"] = fired ? 1 : 0;

        if (target.has_value()) {
          data["target_z"] = target->ekf_x()[4];   //z
          data["target_vz"] = target->ekf_x()[5];  //vz
        }

        if (target.has_value()) {
          data["w"] = target->ekf_x()[7];
        } else {
          data["w"] = 0.0;
        }


        plotter.plot(data);

      
        std::this_thread::sleep_for(10ms);
    }
  });
  std::chrono::steady_clock::time_point last;
  // 主循环
  while (!exiter.exit()) {
    // 读取云台模式
    auto mode = gimbal.mode();
    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last);
    tools::logger()->info("{:.2f} fps", 1 / dt);
    last = now;
    // // 模式切换日志
    // if (last_mode != mode) {
    //   tools::logger()->info("Switch to {}", gimbal.str(mode));
    //   last_mode = mode;
    // }
    
    // 只处理自瞄模式
    if (mode != io::GimbalMode::AUTO_AIM) {
      // 非自瞄模式：发送停止指令并跳过
      std::this_thread::sleep_for(50ms);
      continue;
    }
    
    // 读取主相机图像
    bincameras.cameras.aim_ptr->read(img, timestamp);
    
    // 获取云台姿态（四元数）
    Eigen::Quaterniond q = gimbal.q(timestamp);
    


    // 更新解算器姿态
    bincameras.solvers.aim_ptr-> set_R_gimbal2world(q);
    
    // 获取云台欧拉角
    gimbal_euler = tools::eulers(bincameras.solvers.aim_ptr->R_gimbal2world(), 2, 1, 0);

    float yaw_deg = gimbal_euler[0] * 180.0 / M_PI;
    float pitch_deg = gimbal_euler[1] * 180.0 / M_PI;
    float roll_deg = gimbal_euler[2] * 180.0 / M_PI;

    // std::cout << "DK_Yaw: " << yaw_deg << std::endl;
    // std::cout << "DK_Pitch: " << pitch_deg << std::endl;
    if(yaw_deg == 0 || pitch_deg ==0)std::cout<<"shit"<<std::endl;
     tools::draw_text(img, fmt::format("DK_Yaw {:.2f}", yaw_deg), {40, 40}, {0, 0, 255});
      tools::draw_text(img, fmt::format("DK_Pitch {:.2f}", pitch_deg), {40, 80}, {0, 0, 255});
    // std::cout << "Roll: " << roll_deg << std::endl;

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
      // 【新增】：唤醒全向相机（恢复底层硬件推流）
      omn_cam1.resume();
      omn_cam2.resume();
      if(!bincameras.is_short){
        tools::logger()->info("进入全向感知模式，切换至短焦镜头");
        bincameras.Switch(tracker);
      }

      // 全向感知模式
      io::VisionToGimbal vision_cmd = decider.decide_g(
        yolo, gimbal_euler, omn_cam1, omn_cam2);
        
      gimbal.send(vision_cmd);
    } else {
        // 【新增】：挂起全向相机（停止底层硬件推流，释放CPU和USB/网卡带宽）
        omn_cam1.pause();
        omn_cam2.pause();      
    }

    {
      // 自瞄模式 - 使用MPC
      if (!targets.empty()) {
        auto& target = targets.front();
            // 获取EKF状态向量
        Eigen::VectorXd ekf_x = target.getEKFXest();
        
        // 1. 计算旋转中心的世界坐标
        // EKF状态: [x, vx, y, vy, z, vz, angle, w, r, l, h]
        // 旋转中心: (x, y, z) = (ekf_x[0], ekf_x[2], ekf_x[4])
        Eigen::Vector3d center_world(ekf_x[0], ekf_x[2], ekf_x[4]);
        
        // 2. 计算速度终点（预测0.5秒后的位置）
        double dt = 0.5; // 预测时间
        double scale_factor = 1.0; // 放大2倍
        Eigen::Vector3d velocity(ekf_x[1], ekf_x[3], ekf_x[5]);
        Eigen::Vector3d pred_center = center_world + velocity * dt * scale_factor ;
        
        // 3. 计算角速度方向终点
        double w = ekf_x[7]; // 角速度
        Eigen::Vector3d v_yaw_axis_tvec = center_world;
        v_yaw_axis_tvec[2] += w * 0.1; // 在y方向加上角速度的影响

        double speed_magnitude = std::sqrt(ekf_x[1]*ekf_x[1] + ekf_x[3]*ekf_x[3] + ekf_x[5]*ekf_x[5]);


        // std::cout << "角速度大小: " << w * 57.3 << " °/s" << std::endl;
        // 5. 输出速度大小到控制台
        // std::cout << "速度大小: " << speed_magnitude << " m/s" << std::endl;
        
        // 4. 将世界坐标转换为图像坐标
        // 这里需要将世界坐标转换为相机坐标，然后再投影到图像
        // 假设solver有一个将世界坐标转换为图像坐标的函数
        // 如果没有，你可以创建一个简单的投影函数
        
        // 方法1: 如果solver有直接投影点的函数
        // auto center_img = solver.reproject_point(center_world);
        // auto pred_point_img = solver.reproject_point(pred_center);
        // auto v_yaw_axis_point_img = solver.reproject_point(v_yaw_axis_tvec);
        
        // 方法2: 使用reproject_armor函数（需要一个虚拟的装甲板）
        // 这里假设我们有一个虚拟装甲板用于投影
        auto center_img = bincameras.solvers.aim_ptr->reproject_armor(center_world, 0.0, target.armor_type, target.name);
        auto pred_point_img = bincameras.solvers.aim_ptr->reproject_armor(pred_center, 0.0, target.armor_type, target.name);
        auto v_yaw_axis_point_img = bincameras.solvers.aim_ptr->reproject_armor(v_yaw_axis_tvec, 0.0, target.armor_type, target.name);
        
        // 5. 绘制速度和角速度方向
        if (!center_img.empty() && !pred_point_img.empty()) {
            // 绘制旋转中心
            cv::circle(img, center_img[0], 5, cv::Scalar(51, 153, 237), -1);
            
            // 绘制预测点（速度方向）
            cv::circle(img, pred_point_img[0], 8, cv::Scalar(0, 0, 255), -1);
            
            // 绘制速度方向线
            cv::line(img, center_img[0], pred_point_img[0], cv::Scalar(0, 255, 255), 2);
            
            // 绘制角速度方向线
            if (!v_yaw_axis_point_img.empty()) {
            cv::line(img, center_img[0], v_yaw_axis_point_img[0], cv::Scalar(0, 255, 0), 2);
            }
        }
        // 将目标放入队列供MPC线程处理
        target_queue.push(targets.front());
        bincameras.ChangeTheScope(targets.front(), tracker);
      } else {
        target_queue.push(std::nullopt);
      }


      if (!targets.empty()) {
        auto target = targets.front();

        // 当前帧target更新后
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points =
            bincameras.solvers.aim_ptr->reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        Eigen::Vector4d aim_xyza = bincameras.planners.aim_ptr->debug_xyza;
        auto image_points =
          bincameras.solvers.aim_ptr->reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
     }
    }


    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    // if (key == 'c'){// 强制切换长短焦
    //     bincameras.Switch(tracker);
    // }



    
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