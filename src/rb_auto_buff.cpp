#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/rm_buff_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | ../configs/xiaohei.yaml | 完整车辆配置文件路径}";

void draw_reprojection(
  cv::Mat & image, const std::vector<cv::Point2f> & points, const cv::Scalar & color,
  const std::string & label)
{
  if (points.size() < 4) return;

  std::vector<cv::Point2f> target(points.begin(), points.begin() + 4);
  tools::draw_points(image, target, color, 3);
  if (points.size() > 4) cv::circle(image, points[4], 6, color, 2);
  cv::putText(
    image, label, target.front() + cv::Point2f(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_buff::Rm_Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::SmallTarget target;
  auto_buff::Aimer aimer(config_path);
  detector.set_debug_draw(true);

  constexpr char WINDOW_NAME[] = "small_buff";
  cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
  cv::resizeWindow(WINDOW_NAME, 1280, 720);

  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  auto last_frame_time = std::chrono::steady_clock::now();
  double displayed_fps = 0.0;
  auto last_mode = io::GimbalMode::IDLE;

  while (!exiter.exit()) {
    camera.read(image, timestamp);
    if (image.empty()) continue;

    const auto now = std::chrono::steady_clock::now();
    const double frame_dt = std::chrono::duration<double>(now - last_frame_time).count();
    last_frame_time = now;
    if (frame_dt > 1e-4) {
      const double instant_fps = 1.0 / frame_dt;
      displayed_fps = displayed_fps == 0.0 ? instant_fps : displayed_fps * 0.9 + instant_fps * 0.1;
    }

    const auto mode = gimbal.mode();
    const auto state = gimbal.state();
    if (mode != last_mode) {
      tools::logger()->info("[AutoBuff] mode={}", gimbal.str(mode));
      last_mode = mode;
    }

    detector.set_enemy_color(state.enemy_color);
    solver.set_R_gimbal2world(gimbal.q(timestamp));
    auto power_rune = detector.detect(image, timestamp);
    solver.solve(power_rune);
    target.get_target(power_rune, timestamp);

    auto_aim::Plan plan{};
    if (mode == io::GimbalMode::SMALL_BUFF && !target.is_unsolve()) {
      auto predicted_target = target;
      plan = aimer.mpc_aim(predicted_target, timestamp, state, true);
      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      const auto center = target.point_buff2world(Eigen::Vector3d::Zero());
      const auto aim_points =
        solver.reproject_buff(center, predicted_target.ekf_x()[4], predicted_target.ekf_x()[5]);
      draw_reprojection(image, aim_points, cv::Scalar(0, 0, 255), "BALLISTIC");
    } else {
      gimbal.send(false, false, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
    }

    const cv::Scalar mode_color =
      mode == io::GimbalMode::SMALL_BUFF ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::putText(
      image, fmt::format("MODE {}", gimbal.str(mode)), cv::Point(10, 88), cv::FONT_HERSHEY_SIMPLEX,
      0.65, mode_color, 2);
    cv::putText(
      image,
      fmt::format(
        "FPS {:.1f} DETECT {} CONTROL {} FIRE {}", displayed_fps,
        power_rune.has_value() ? "OK" : "LOST", plan.control ? "ON" : "OFF",
        plan.fire ? "ON" : "OFF"),
      cv::Point(10, 116), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      power_rune.has_value() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 2);

    cv::imshow(WINDOW_NAME, image);
    const int key = cv::waitKey(1) & 0xff;
    if (key == 'q' || key == 27) break;
  }

  gimbal.send(false, false, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);
  cv::destroyAllWindows();
  return 0;
}
