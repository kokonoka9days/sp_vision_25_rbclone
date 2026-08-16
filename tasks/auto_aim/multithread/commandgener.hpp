#ifndef AUTO_AIM_MULTITHREAD__HPP
#define AUTO_AIM_MULTITHREAD__HPP

#include <optional>

#include "io/cboard.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/plotter.hpp"

namespace auto_aim
{
namespace multithread
{

class CommandGener
{
public:
  /** @brief 绑定命令生成所需组件并启动工作线程 @param shooter 开火判定器 @param aimer 瞄准器 @param cboard 控制板 @param plotter 调试绘图器 @param debug 是否发送调试数据 */
  CommandGener(
    auto_aim::Shooter & shooter, auto_aim::Aimer & aimer, io::CBoard & cboard,
    tools::Plotter & plotter, bool debug = false);

  /** @brief 停止并等待命令生成线程 */
  ~CommandGener();

  /** @brief 提交最新跟踪结果供异步生成命令 @param targets 目标列表 @param t 帧时间戳 @param bullet_speed 弹速 @param gimbal_pos 云台位置 */
  void push(
    const std::list<auto_aim::Target> & targets, const std::chrono::steady_clock::time_point & t,
    double bullet_speed, const Eigen::Vector3d & gimbal_pos);

private:
  struct Input
  {
    std::list<auto_aim::Target> targets_;
    std::chrono::steady_clock::time_point t;
    // std::function<void()> decide;
    double bullet_speed;
    Eigen::Vector3d gimbal_pos;
  };

  io::CBoard & cboard_;
  auto_aim::Shooter & shooter_;
  auto_aim::Aimer & aimer_;
  tools::Plotter & plotter_;

  std::optional<Input> latest_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::thread thread_;
  bool stop_, debug_;

  /** @brief 命令生成工作线程入口 */
  void generate_command();
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM_MULTITHREAD__HPP
