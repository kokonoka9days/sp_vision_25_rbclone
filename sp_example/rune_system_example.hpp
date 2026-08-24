#ifndef SP_EXAMPLE__RUNE_SYSTEM_EXAMPLE_HPP
#define SP_EXAMPLE__RUNE_SYSTEM_EXAMPLE_HPP

#include <string>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera/camera.hpp"
#include "io/gimbal/cboard.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/rune_debug_draw.hpp"
#include "tasks/auto_buff/rune_system.hpp"
#include "tools/exiter.hpp"

namespace sp_example
{
inline auto_buff::EnemyColor configured_enemy_color(const std::string & config_path)
{
  const YAML::Node yaml = YAML::LoadFile(config_path);
  const std::string value = yaml["enemy_color"] ? yaml["enemy_color"].as<std::string>() : "blue";
  return value == "red" ? auto_buff::EnemyColor::RED : auto_buff::EnemyColor::BLUE;
}

inline int run_gimbal_rune_example(int argc, char ** argv, bool debug)
{
  const std::string keys =
    "{help h usage ? | | show help }"
    "{@config-path | ../configs/xiaohei.yaml | robot yaml }";
  cv::CommandLineParser cli(argc, argv, keys);
  const std::string config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_buff::RuneSystem rune(config_path);
  const auto color = configured_enemy_color(config_path);
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  while (!exiter.exit()) {
    camera.read(image, timestamp);
    const auto mode = gimbal.mode();
    if (mode != io::GimbalMode::SMALL_BUFF && mode != io::GimbalMode::BIG_BUFF) {
      rune.reset();
      gimbal.send(false, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
      continue;
    }
    const auto state = gimbal.state();
    const auto command = rune.process(
      image, timestamp, gimbal.q(timestamp),
      mode == io::GimbalMode::BIG_BUFF ? auto_buff::BuffMode::BIG : auto_buff::BuffMode::SMALL,
      color, state.bullet_speed);
    gimbal.send(
      command.found, command.fire, command.yaw, 0.0f, 0.0f,
      command.pitch, 0.0f, 0.0f);
    if (debug) {
      auto_buff::draw_rune_debug(image, rune.debug_snapshot());
      cv::imshow("RuneSystem", image);
      if (cv::waitKey(1) == 'q') break;
    }
  }
  return 0;
}

inline int run_cboard_rune_example(int argc, char ** argv, bool debug)
{
  const std::string keys =
    "{help h usage ? | | show help }"
    "{@config-path | ../configs/demo.yaml | robot yaml }";
  cv::CommandLineParser cli(argc, argv, keys);
  const std::string config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  io::Camera camera(config_path);
  io::CBoard cboard(config_path);
  auto_buff::RuneSystem rune(config_path);
  const auto color = configured_enemy_color(config_path);
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  while (!exiter.exit()) {
    camera.read(image, timestamp);
    if (cboard.mode != io::Mode::small_buff && cboard.mode != io::Mode::big_buff) {
      rune.reset();
      cboard.send({false, false, 0.0, 0.0});
      continue;
    }
    const auto command = rune.process(
      image, timestamp, cboard.imu_at(timestamp),
      cboard.mode == io::Mode::big_buff ? auto_buff::BuffMode::BIG : auto_buff::BuffMode::SMALL,
      color, static_cast<float>(cboard.bullet_speed));
    cboard.send({command.found, command.fire, command.yaw, command.pitch});
    if (debug) {
      auto_buff::draw_rune_debug(image, rune.debug_snapshot());
      cv::imshow("RuneSystem", image);
      if (cv::waitKey(1) == 'q') break;
    }
  }
  return 0;
}
}  // namespace sp_example

#endif
