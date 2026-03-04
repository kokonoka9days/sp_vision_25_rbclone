#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp" 
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
  "{help h usage ? |                                             | 输出命令行参数说明}"
  "{short_camera   | ../configs/sb.yaml                          | 短焦相机配置文件路径 }"
  "{long_camera    | ../configs/sb_copy.yaml                     | 长焦相机配置文件路径 }"
  "{l_cam          | ../configs/omniperception/short_camera.yaml | 左感知相机 }"
  "{r_cam          | ../configs/omniperception/long_camera.yaml  | 右感知相机 }";

using namespace std::chrono_literals;


int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  int i =0;
  
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto short_camera_config_path = cli.get<std::string>("short_camera");
  auto long_camera_config_path = cli.get<std::string>("long_camera");

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
  tools::logger()->info("初始化");
  // 改为使用Gimbal串口通信（替代CBoard）
  // io::Gimbal gimbal(short_camera_config_path);
  
  // 视觉模块
  auto_aim::YOLO yolo(short_camera_config_path, false);  // 主相机YOLO

  auto_aim::Solver short_camera_solver(short_camera_config_path);
  auto_aim::Solver long_camera_solver(long_camera_config_path);

  // auto_aim::Tracker tracker(short_camera_config_path, short_camera_solver);//默认短焦
  // tracker.set_gimbal(&gimbal);
  auto_aim::Aimer aimer(short_camera_config_path);
  auto_aim::Shooter shooter(short_camera_config_path);
  
  
  cv::Mat img1, img2, img3, img4;
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point last_t;
  std::chrono::steady_clock::time_point last_t1;
  
  // 获取云台模式
  auto last_mode = io::GimbalMode::IDLE;

  // 新增一个变量用于记录全向相机是否处于暂停状态
  bool is_omn_paused = false; 

  // 主循环
  while (!exiter.exit()) {
    // 读取云台模式
    // auto mode = gimbal.mode();
    
    // 如果没有暂停，才去读取全向相机图像
    if (!is_omn_paused) {
        omn_cam1.read(img1, timestamp);
        omn_cam2.read(img2, timestamp);
        long_camera.read(img4, timestamp);
    }
    
    short_camera.read(img3, timestamp);
    // long_camera.read(img4, timestamp);

    // auto now = std::chrono::steady_clock::now();
    // auto dt = tools::delta_time(now, last_t);
    // tools::logger()->info("{:.2f} fps", 1 / dt);
    // last_t = now;

    // nlohmann::json data;
    //   data["t"] = 1 / dt;

    //   plotter.plot(data);

    i++;
    if(i == 100)
    {
      i = 0;
      auto now1 = std::chrono::steady_clock::now();
      auto dt1 = tools::delta_time(now1, last_t1);
      last_t1 = now1;
      tools::logger()->info("100fps_time:{:.2f} s", dt1);
    }


    // 同样，如果没有暂停才更新显示（避免显示空矩阵）
    if (!is_omn_paused && !img1.empty() && !img2.empty()) {
        // cv::resize(img1, img1, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
        cv::imshow("omn_cam1", img1);
        cv::imshow("omn_cam2", img2);
        cv::imshow("long_camera", img4);
    }
    
    cv::imshow("short_camera", img3);
    // cv::imshow("long_camera", img4);
    
    auto key = cv::waitKey(1);
    if (key == 'q') break;
    
    if( key == 'p') {
      // 暂停全向相机
      tools::logger()->info("omn_cam stop");
      omn_cam1.pause();
      omn_cam2.pause();
      long_camera.pause();
      is_omn_paused = true; // 更新状态标志
    }
    if( key == 'r') {
      // 恢复全向相机
      last_t = std::chrono::steady_clock::now();
      long_camera.resume();
      omn_cam1.resume();
      omn_cam2.resume();
      is_omn_paused = false; // 更新状态标志
      auto now = std::chrono::steady_clock::now();
      auto dt = tools::delta_time(now, last_t);
      tools::logger()->info("{:.2f} s",dt);
    }
}
}