#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{

enum class WorkMode : uint8_t
{
  IDLE = 0,              // 空闲
  AUTO_AIM = 1,          // 自瞄模式
  OMNI_PERCEPTION = 2,   // 全向感知模式
};

struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {0x5a,0x53};
  uint8_t mode;  // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符, 4:开长焦  电控控制右键0，1
  uint16_t color; // 0: 红色, 1: 蓝色
  float q[4];    // wxyz顺序
  float bullet_speed;
  uint16_t bullet_count;  // 子弹累计发送次数
  float gimbal_yaw;
  float gimbal_pitch;
  uint16_t crc16;
};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head = {0x66};
  uint8_t mode = 0;  // 0: 不控制, 1: 控制云台但不开火，2:控制云台开火                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  uint16_t crc16;
  uint8_t end = {0x11};
};

struct __attribute__((packed)) sb_VisionToGimbal
{
  uint8_t head = {0x66};
  uint8_t mode = 0;  // 0: 不控制, 1: 控制云台但不开火，2:控制云台开火                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  float target_x = 0;
  float target_y = 0;
  uint8_t target_name = 0;
  uint8_t end = {0x11};
};

struct __attribute__((packed)) OmniVisionToGimbal
{
  uint8_t head = {0x66};
  uint8_t mode = 0;
  float yaw = 0;       // rad
  float pitch = 0;     // rad
  float distance = 0;  // m, target-to-gimbal 3D distance
  uint8_t end = {0x11};
};

static_assert(sizeof(VisionToGimbal) <= 64);
static_assert(sizeof(OmniVisionToGimbal) == 15);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF,     // 大符
  LONG_FOCAL_LENGTH //长焦
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float q2yaw;
  float q2pitch;
  uint8_t mode;
  uint8_t enemy_color; // 0: 红色, 1: 蓝色
  float bullet_speed;
  uint16_t bullet_count;
};

class Gimbal
{
public:
  /** @brief 根据配置文件初始化云台串口通信 @param config_path YAML 配置文件路径 */
  Gimbal(const std::string & config_path);

  /** @brief 停止接收线程并关闭串口 */
  ~Gimbal();

  /** @brief 获取当前云台工作模式 @return 云台模式 */
  GimbalMode mode() const;
  /** @brief 获取线程安全的云台状态快照 @return 云台状态 */
  GimbalState state() const;
  /** @brief 将云台模式转换为字符串 @param mode 云台模式 @return 模式名称 */
  std::string str(GimbalMode mode) const;
  /** @brief 插值得到指定时刻的云台姿态 @param t 查询时间戳 @return 姿态四元数 */
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  /** @brief 发送自瞄控制量 @param control 是否接管控制 @param fire 是否开火 @param yaw 目标偏航角 @param yaw_vel 偏航角速度 @param yaw_acc 偏航角加速度 @param pitch 目标俯仰角 @param pitch_vel 俯仰角速度 @param pitch_acc 俯仰角加速度 */
  void send(
    bool control,bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  
  /** @brief 发送哨兵自瞄控制量 @param control 是否接管控制 @param fire 是否开火 @param yaw 目标偏航角 @param yaw_vel 偏航角速度 @param yaw_acc 偏航角加速度 @param pitch 目标俯仰角 @param pitch_vel 俯仰角速度 @param pitch_acc 俯仰角加速度 @param target_x 目标横坐标 @param target_y 目标纵坐标 @param target_name 目标编号 */
  void sb_send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc,
  float pitch, float pitch_vel, float pitch_acc, float target_x, float target_y, uint8_t target_name);

  /** @brief 发送已组装的自瞄数据帧 @param VisionToGimbal 数据帧 */
  void send(io::VisionToGimbal VisionToGimbal);

  /** @brief 发送已组装的哨兵自瞄数据帧 @param VisionToGimbal 数据帧 */
  void sb_send(io::sb_VisionToGimbal VisionToGimbal);

  /** @brief 发送全向感知目标 @param mode 控制模式 @param yaw 目标偏航角 @param pitch 目标俯仰角 @param distance 目标距离，单位 m */
  void omni_send(uint8_t mode, float yaw, float pitch, float distance);

  /** @brief 发送已组装的全向感知数据帧 @param VisionToGimbal 数据帧 */
  void omni_send(const io::OmniVisionToGimbal & VisionToGimbal);

  /** @brief 获取内部状态的可写指针 @return 云台状态指针 */
  GimbalState* set_state_(){
    return &state_;
  }

private:
  serial::Serial serial_;
  std::vector<std::string> com_ports_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_;
  VisionToGimbal tx_data_;
  sb_VisionToGimbal sb_tx_data_;
  OmniVisionToGimbal omni_tx_data_;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  int gimbal_yaw2vision, gimbal_pitch2vision, gimbal_roll2vision;

  /** @brief 从串口读取指定字节数 @param buffer 输出缓冲区 @param size 期望字节数 @return 读取成功时返回 true */
  bool read(uint8_t * buffer, size_t size);
  /** @brief 尝试打开配置的串口 @return 打开成功时返回 true */
  bool open_serial();
  /** @brief 云台串口接收线程入口 */
  void read_thread();
  /** @brief 循环尝试重新连接串口 */
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
