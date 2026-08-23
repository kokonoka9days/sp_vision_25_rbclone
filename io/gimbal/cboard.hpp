#ifndef IO__CBOARD_HPP
#define IO__CBOARD_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "io/socketcan.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
enum Mode
{
  idle,
  auto_aim,
  small_buff,
  big_buff,
  outpost
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

// 哨兵专有
enum ShootMode
{
  left_shoot,
  right_shoot,
  both_shoot
};
const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

class CBoard
{
public:
  double bullet_speed;
  Mode mode;
  ShootMode shoot_mode;
  double ft_angle;  //无人机专有

  /** @brief 根据配置文件初始化控制板 CAN 通信 @param config_path YAML 配置文件路径 */
  CBoard(const std::string & config_path);

  /** @brief 插值得到指定时刻的 IMU 姿态 @param timestamp 查询时间戳 @return 姿态四元数 */
  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  /** @brief 向控制板发送云台控制命令 @param command 控制命令 */
  void send(Command command) const;

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  tools::ThreadSafeQueue<IMUData> queue_;  // 必须在can_之前初始化，否则存在死锁的可能
  SocketCAN can_;
  IMUData data_ahead_;
  IMUData data_behind_;

  int quaternion_canid_, bullet_speed_canid_, send_canid_;

  /** @brief 处理接收到的 CAN 帧 @param frame CAN 数据帧 */
  void callback(const can_frame & frame);

  /** @brief 读取控制板配置 @param config_path YAML 配置文件路径 @return CAN 接口名称 */
  std::string read_yaml(const std::string & config_path);
};

}  // namespace io

#endif  // IO__CBOARD_HPP
