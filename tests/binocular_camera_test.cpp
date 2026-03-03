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
#include "tools/img_tools.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{short_camera   | ../configs/sb.yaml | 短焦相机配置文件路径 }"
  "{long_camera    | ../configs/sb_copy.yaml  | 长焦相机配置文件路径 }";

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
  
//   串口通信
//   io::Gimbal gimbal(short_camera_config_path);
  
  // 视觉模块
  auto_aim::YOLO yolo(short_camera_config_path, false);  // 主相机YOLO

  auto_aim::Solver short_camera_solver(short_camera_config_path);
  auto_aim::Solver long_camera_solver(long_camera_config_path);

  auto_aim::Tracker tracker(short_camera_config_path, short_camera_solver);//默认短焦
  
  // MPC 规划器
  auto_aim::Planner short_camera_planner(short_camera_config_path);
  auto_aim::Planner long_camera_planner(long_camera_config_path);

  //双目切换
  BinocularAim bincameras(short_camera, long_camera, 
                          short_camera_solver, long_camera_solver, 
                          short_camera_planner, long_camera_planner );
  
  
  // 线程安全队列（用于MPC规划线程）
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);
  
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point last_t;
  
  // 云台状态
  Eigen::Vector3d gimbal_euler;
  
  // 获取云台模式
  auto last_mode = io::GimbalMode::IDLE;
  
  // MPC 规划线程（独立线程运行MPC控制器）
  std::atomic<bool> quit = false;

  // 主循环
  while (!exiter.exit()) {
    static int count = 0;
    // tools::logger()->info("当前使用 {} 焦镜头", bincameras.is_short ? "短" : "长");
    // // 读取主相机图像
    bincameras.cameras.aim_ptr->read(img, timestamp);
    
    
    // // 获取云台欧拉角
    // gimbal_euler = tools::eulers(bincameras.solvers.aim_ptr->R_gimbal2world(), 2, 1, 0);
    
    // 主相机检测
    auto armors = yolo.detect(img);
    
    // 跟踪目标
    auto targets = tracker.track(armors, timestamp);
    double fps = 1./std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - last_t).count()*1000000;
    tools::draw_text(img, "fps: "+std::to_string(fps), cv::Point(40, 130));
    last_t = std::chrono::steady_clock::now();
    tools::logger()->info("fps:: {:.2f}", fps);
     
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

    
    // cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    // cv::imshow("reprojection", img);
    // auto key = cv::waitKey(1);
    // if (key == 'q') break;
    // if (key == 'c'){// 强制切换
    //     bincameras.Switch();
    // }
  }
  
  // 清理
  quit = true;
  
  
  return 0;
}