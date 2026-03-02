#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"  // 改为使用Gimbal串口通信
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
  "{short_camera   | ../configs/sb.yaml | 短焦相机配置文件路径 }"
  "{long_camera    | ../configs/sb_copy.yaml  | 长焦相机配置文件路径 }"
  "{l_cam          | ../configs/omniperception/short_camera.yaml  | 左感知相机 }"
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
    aim_ptr = aim_ptr == &short_aim ? &long_aim : &short_aim;
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
  double short2long_point =  3.0;//(short_max_far + long_min_near)/2.;
  double long2short_point = 1.7;

  /// @brief 长短焦切换
  void Switch(){
    this->cameras.Switch();
    this->solvers.Switch();
    this->planners.Switch();
    is_short = !is_short;
    switch_time_point = std::chrono::steady_clock::now();
  }

  /// @brief 长短焦切换逻辑
  void ChangeTheScope(auto_aim::Target target , auto_aim::Tracker& tracker){
    const auto x_est = target.getEKFXest();
    const double x = x_est(0), y = x_est(2), z = x_est(4);
    double dis = sqrt( x*x + y*y + z*z);
    
    if(is_short && dis > short2long_point ){
      tools::logger()->info("切换至长焦镜头");
      Switch();
    }else if(!is_short && dis < long2short_point){
      tools::logger()->info("切换至短焦镜头");
      Switch();
    }

    tracker.setSolver(*this->solvers.aim_ptr);
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

  auto_aim::Tracker tracker(short_camera_config_path, short_camera_solver);//默认短焦
  tracker.set_gimbal(&gimbal);
  auto_aim::Aimer aimer(short_camera_config_path);
  auto_aim::Shooter shooter(short_camera_config_path);
  
  
  // 全向感知决策器
  omniperception::Decider decider(short_camera_config_path);
  
  // 线程安全队列（用于MPC规划线程）
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);
  
  cv::Mat img1, img2, img3, img4;
  std::chrono::steady_clock::time_point timestamp;
  
  // 云台状态
  Eigen::Vector3d gimbal_euler;
  
  // 获取云台模式
  auto last_mode = io::GimbalMode::IDLE;


  // 主循环
  while (!exiter.exit()) {
    // 读取云台模式
    auto mode = gimbal.mode();
    omn_cam1.read(img1, timestamp);
    omn_cam2.read(img2, timestamp);
    short_camera.read(img3, timestamp);
    long_camera.read(img4, timestamp);

    // cv::resize(img1, img1, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("omn_cam1", img1);
    cv::imshow("omn_cam2", img2);
    cv::imshow("short_camera", img3);
    cv::imshow("long_camera", img4);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }
  
  // 发送停止指令
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  
  return 0;
}