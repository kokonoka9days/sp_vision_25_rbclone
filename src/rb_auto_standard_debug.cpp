#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/recorder.hpp"

// 打符相关头文件
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"

using namespace std::chrono_literals;
using namespace tools;


const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | ../configs/xiaohuang.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 自瞄相关对象初始化
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, &solver);
  tracker.set_gimbal(&gimbal);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);
  // tools::Recorder recor(90);
  bool stopkey = false;

  // 打符相关对象初始化
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE}; // 全局云台模式

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      // 【修改】非打符模式，全认为是自瞄
      if (mode.load() != io::GimbalMode::SMALL_BUFF && mode.load() != io::GimbalMode::BIG_BUFF) {
        auto target = target_queue.front(); 
        auto gs = gimbal.state();

        // MPC预测以及+自家火控
        auto plan = planner.plan(target, gs.bullet_speed, gs.yaw,  auto_aim::Planner::ShootStrategy::rbSuppressiveFire);

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
        data["plan_yaw"] = plan.yaw / CV_PI * 180. ;
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
          data["ekf_vyaw"] = ekf_satic(7) * 57.3;
          data["ekf_r"] = ekf_satic(8);        
        }

        plotter.plot(data);

        std::this_thread::sleep_for(10ms);
      } else {
        // 若是打符模式，由主循环发送指令，发送线程仅休眠
        std::this_thread::sleep_for(10ms);
      }
    }
  });

  auto t0_main = std::chrono::steady_clock::now();
  uint16_t last_bullet_count_main = 0;

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point last_t;
  auto last_mode{io::GimbalMode::IDLE}; // 记录上次模式

  while (!exiter.exit()) {
    mode = gimbal.mode(); // 每帧获取最新云台模式
    
    if (last_mode != mode.load()) {
      tools::logger()->info("Switch to {}", gimbal.str(mode.load()));
      last_mode = mode.load();
    }

    camera.read(img, t);
    auto q = gimbal.q(t - 3ms);

    double fps = 1./std::chrono::duration_cast<std::chrono::microseconds>(t - last_t).count()*1000000;
    // tools::draw_text(img, "fps: "+std::to_string(fps), cv::Point(40, 130));
    last_t = t;
    // tools::logger()->info("fps:: {:.2f}", fps);

    auto ypr = tools::eulers(q, 2, 1, 0);

    float yaw_deg = ypr[0] * 180.0 / M_PI;
    float pitch_deg = ypr[1] * 180.0 / M_PI;
    float roll_deg = ypr[2] * 180.0 / M_PI;
        
    // std::cout << "DK_Yaw: " << yaw_deg << std::endl;
    // std::cout << "DK_Pitch: " << pitch_deg << std::endl;
    // if(yaw_deg == 0 || pitch_deg ==0)std::cout<<"shit"<<std::endl;
    // tools::draw_text(img, fmt::format("rb_Yaw {:.2f}", yaw_deg), {40, 40}, {0, 128, 255});
    // tools::draw_text(img, fmt::format("rb_Pitch {:.2f}", pitch_deg), {40, 80}, {0, 255, 255});
    // std::cout << "Roll: " << roll_deg << std::endl;

    // 【修改】如果是打符模式
    if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      auto gs = gimbal.state();
      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);

      buff_solver.solve(power_runes);

      auto_aim::Plan buff_plan;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      }
      
      // 直接发送打符相关的控制指令
      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

      // ================== 绘制打符的预测目标框 ==================
      // 确保当前识别到了能量机关，且位姿解算成功
      if (power_runes.has_value() && !power_runes->is_unsolve()) {
        
        // buff_aimer.angle 存储了预测后的扇叶绝对角度 (对应 solver 中的 row，即 roll)
        double pred_angle = buff_aimer.angle; 
        
        // 调用打符专属的重投影函数
        auto image_points = buff_solver.reproject_buff(
            power_runes->xyz_in_world,     // 大符圆心的世界坐标
            power_runes->ypr_in_world[0],  // 大符平面的 yaw 偏角
            pred_angle                     // 预测后的扇叶角度
        );

        // buff_solver 中的 OBJECT_POINTS 包含了7个点，前4个为装甲板四角
        if (image_points.size() >= 4) {
          // 提取装甲板4个角点并绘制红框
          std::vector<cv::Point2f> armor_pts(image_points.begin(), image_points.begin() + 4);
          tools::draw_points(img, armor_pts, {0, 0, 255}, 2); // 红色预测框

          if (image_points.size() >= 7) {
            // image_points[4]: 装甲板中心 (0, 0, 700e-3)
            // image_points[6]: 能量机关 R 标中心 (0, 0, 0)
            
            // 绘制预测的装甲板中心点 (黄色)
            cv::circle(img, image_points[4], 6, cv::Scalar(0, 255, 255), -1);
            
            // 绘制能量机关 R 标中心点 (紫色)
            cv::circle(img, image_points[6], 6, cv::Scalar(255, 0, 255), -1);
            
            // 绘制 R标 到 装甲板中心 的预测连线 (绿色，用来直观显示预测后的扇叶悬臂位置)
            cv::line(img, image_points[6], image_points[4], cv::Scalar(0, 255, 0), 2);
          }
        }
      }
      // =======================================================

      auto fired = gs.bullet_count > last_bullet_count_main;
      last_bullet_count_main = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0_main);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = buff_plan.target_yaw;
      data["target_pitch"] = buff_plan.target_pitch;

      data["plan_mode"] = buff_plan.control ? (buff_plan.fire ? 2 : 1) : 0;
      data["plan_yaw"] = buff_plan.yaw / CV_PI * 180.;
      data["plan_yaw_vel"] = buff_plan.yaw_vel;
      data["plan_yaw_acc"] = buff_plan.yaw_acc;

      data["plan_pitch"] = buff_plan.pitch * 57.3;
      data["plan_pitch_vel"] = buff_plan.pitch_vel;
      data["plan_pitch_acc"] = buff_plan.pitch_acc;

      data["fire"] = buff_plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;


      plotter.plot(data);
    }
    // 【修改】除打符模式外，全认为是自瞄
    else {
      // 原版自瞄逻辑
      solver.set_R_gimbal2world(q);
      auto armors = yolo.detect(img);
      auto targets = tracker.track(armors, t);
      // recor.record(img, q, t);

      if (!targets.empty()){
        target_queue.push(targets.front());

        auto& target = targets.front();
      
        // 获取EKF状态向量
        Eigen::VectorXd ekf_x = target.getEKFXest();
        
        // 1. 计算旋转中心的世界坐标
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

        auto center_img = solver.reproject_armor(center_world, 0.0, target.armor_type, target.name);
        auto pred_point_img = solver.reproject_armor(pred_center, 0.0, target.armor_type, target.name);
        auto v_yaw_axis_point_img = solver.reproject_armor(v_yaw_axis_tvec, 0.0, target.armor_type, target.name);
        
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
      }
      else {
        target_queue.push(std::nullopt);
      }

      if (!targets.empty()) {
        auto target = targets.front();

        // 当前帧target更新后
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {235, 206, 135});
        }

        Eigen::Vector4d aim_xyza = planner.debug_xyza;
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
      }

    } 
    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    if(key == 'r') {//TUDO :右键手动更改
      io::GimbalState* g_demo = gimbal.set_state_();
      g_demo->mode = !g_demo->mode;
    }
    // if(key == 's') {
    //   stopkey = !stopkey;
    // }
  }
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  
  // 获取当前下位机发来的云台状态数据
  auto current_state = gimbal.state();
  
  // 发送当前数据（注意：由于 gimbal.cpp 中接收时乘了 57.3 转成了角度，发回下位机时需要除以 57.3 转回弧度）
  // 因为下位机没有发来速度和加速度数据，所以 vel 和 acc 继续填 0 即可
  gimbal.send(
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