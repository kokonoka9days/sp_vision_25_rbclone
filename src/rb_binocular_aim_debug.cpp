#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>


#include "io/gimbal/gimbal.hpp"  
// #include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"
#include "tools/img_tools.hpp"

#include "method_set/binocular_aim.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{short_camera   | ../configs/sb.yaml | 短焦相机配置文件路径 }"
  "{long_camera    | ../configs/sb_copy.yaml  | 长焦相机配置文件路径 }";

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
  auto short_camera_config_path = cli.get<std::string>("short_camera");
  auto long_camera_config_path = cli.get<std::string>("long_camera");


  
  // 主相机（工业相机）
  io::Camera short_camera(short_camera_config_path);
  // io::Camera long_camera(long_camera_config_path);
  
  // 串口通信
  io::Gimbal gimbal(short_camera_config_path);
  
  // 视觉模块
  auto_aim::YOLO yolo(short_camera_config_path, false);  // 主相机YOLO

  auto_aim::Solver short_camera_solver(short_camera_config_path);
  auto_aim::Solver long_camera_solver(long_camera_config_path);

  auto_aim::Tracker tracker(short_camera_config_path, &short_camera_solver);//默认短焦
  tracker.set_gimbal(&gimbal);
  
  // MPC 规划器
  auto_aim::Planner short_camera_planner(short_camera_config_path);
  auto_aim::Planner long_camera_planner(long_camera_config_path);

  //双目切换
  // BinocularAim bincameras(short_camera, long_camera, 
  //                         short_camera_solver, long_camera_solver, 
  //                         short_camera_planner, long_camera_planner );
  
  
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
    auto_aim::Plan plan{false};
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        
        // // 使用MPC规划器计算控制指令
        // plan = bincameras.planners.aim_ptr->plan(*target, 22);
          
        //MPC预测以及+自家火控
        // auto_aim::Planner * plan_short_or_long = target->cam_is_short ? &bincameras.planners.short_aim : &bincameras.planners.long_aim;
        auto_aim::Planner * plan_short_or_long = &short_camera_planner;
        auto plan =  plan_short_or_long->plan(target, gs.bullet_speed, gs.yaw,  auto_aim::Planner::ShootStrategy::rbSuppressiveFire);
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        auto fired = gs.bullet_count > last_bullet_count;
        last_bullet_count = gs.bullet_count;
        nlohmann::json data;
        data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);


        data["gimbal_yaw"] = gs.yaw;
        data["gimbal_yaw_vel"] = gs.yaw_vel;
        data["gimbal_pitch"] = gs.pitch;
        data["gimbal_pitch_vel"] = gs.pitch_vel;

        data["target_yaw"] = plan.target_yaw;
        data["target_pitch"] = plan.target_pitch;

        data["plan_mode"] = plan.control ? (plan.fire ? 2 : 1) : 0;
        data["plan_yaw"] = plan.yaw * 57.3;
        data["plan_yaw_vel"] = plan.yaw_vel;
        data["plan_yaw_acc"] = plan.yaw_acc;

        data["plan_pitch"] = plan.pitch * 57.3;
        data["plan_pitch_vel"] = plan.pitch_vel;
        data["plan_pitch_acc"] = plan.pitch_acc;

        data["fire"] = plan.fire ? 1 : 0;
        data["fired"] = fired ? 1 : 0;

        if (target.has_value()) {
          data["target_z"] = target->ekf_x()[4];   //z
          data["target_vz"] = target->ekf_x()[5];  //vz
          data["tower_h1"] = target->tower_armor_hs[0];
          data["tower_h2"] = target->tower_armor_hs[1];
          data["tower_h3"] = target->tower_armor_hs[2];
          data["tower_armor_h"] = target->tower_armor_h;

          const auto ekf_satic = target->ekf_x();
          data["ekf_x"] = ekf_satic(0);
          data["ekf_vx"] = ekf_satic(1);
          data["ekf_y"] = ekf_satic(2);
          data["ekf_vy"] = ekf_satic(3);
          data["ekf_z"] = ekf_satic(4);
          data["ekf_vz"] = ekf_satic(5);
          data["ekf_yaw"] = ekf_satic(6) * 57.3;
          data["ekf_vyaw"] = ekf_satic(7);
          data["ekf_r"] = ekf_satic(8);        
        }



        plotter.plot(data);

      
        std::this_thread::sleep_for(10ms);
    }
  });

  // 主循环
  while (!exiter.exit()) {

    // 读取主相机图像
    // bincameras.cameras.aim_ptr->read(img, timestamp);
    short_camera.read(img, timestamp);

    // auto q = gimbal.q(timestamp - bincameras.cameras.aim_ptr->timestamp_offset);
    auto q = gimbal.q(timestamp - short_camera.timestamp_offset);


    // tools::logger()->info("当前使用 {} 焦镜头", bincameras.is_short ? "短" : "长");
    
    
    // 获取云台欧拉角
    gimbal_euler = tools::eulers(short_camera_solver.R_gimbal2world(), 2, 1, 0);

    float yaw_deg = gimbal_euler[0] * 180.0 / M_PI;
    float pitch_deg = gimbal_euler[1] * 180.0 / M_PI;
    float roll_deg = gimbal_euler[2] * 180.0 / M_PI;

    // std::cout << "DK_Yaw: " << yaw_deg << std::endl;
    // std::cout << "DK_Pitch: " << pitch_deg << std::endl;
    if(yaw_deg == 0 || pitch_deg ==0)std::cout<<"shit"<<std::endl;
     tools::draw_text(img, fmt::format("DK_Yaw {:.2f}", yaw_deg), {40, 40}, {0, 0, 255});
      tools::draw_text(img, fmt::format("DK_Pitch {:.2f}", pitch_deg), {40, 80}, {0, 0, 255});
    // std::cout << "Roll: " << roll_deg << std::endl;

    // bincameras.solvers.aim_ptr->set_R_gimbal2world(q);short_camera
    short_camera_solver.set_R_gimbal2world(q);
    // 主相机检测
    auto armors = yolo.detect(img);
    
    // 跟踪目标
    
    auto targets = tracker.track(armors, timestamp);
     
    // 自瞄模式 - 使用MPC
    if (!targets.empty()) {
        // 将目标放入队列供MPC线程处理
        target_queue.push(targets.front());
        
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
        auto center_img = short_camera_solver.reproject_armor(center_world, 0.0, target.armor_type, target.name);
        auto pred_point_img = short_camera_solver.reproject_armor(pred_center, 0.0, target.armor_type, target.name);
        auto v_yaw_axis_point_img = short_camera_solver.reproject_armor(v_yaw_axis_tvec, 0.0, target.armor_type, target.name);
        
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
        // 自动长短焦切换
        // bincameras.ChangeTheScope(targets.front(), tracker);
    } else {
        target_queue.push(std::nullopt);
    }


    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          short_camera_solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = short_camera_planner.debug_xyza;
      auto image_points =
        short_camera_solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    // if (key == 'c'){// 强制切换长短焦
    //     bincameras.Switch(tracker);
    // }
  }
  
  // 清理
  quit = true;
  if (mpc_thread.joinable()) {
    mpc_thread.join();
  }
  
  
  return 0;
}