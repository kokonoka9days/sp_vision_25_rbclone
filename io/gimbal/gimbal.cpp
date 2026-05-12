#include "gimbal.hpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include <opencv2/opencv.hpp>

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  this->gimbal_yaw2vision = tools::read<int>(yaml, "gimbal_y1");
  this->gimbal_pitch2vision = tools::read<int>(yaml, "gimbal_p2");
  this->gimbal_roll2vision = tools::read<int>(yaml, "gimbal_r3");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(460800);
    auto timeout = serial::Timeout::simpleTimeout(2); 
    serial_.setTimeout(timeout);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    // 1. 阻塞等待并弹出队列中最老的一帧数据
    auto [q_a, t_a] = queue_.pop();

    // 2. 防死锁核心：如果弹出后队列空了，绝对不能再去调用 front()！
    // 否则 front() 会永久阻塞等待下一个数据导致画面卡死。
    // 此时直接返回当前唯一可用的数据即可。
    // if (queue_.empty()) {
    //   return q_a;
    // }

    // 3. 此时队列非空，可以安全地偷看（不弹出）下一个数据，绝不会阻塞
    auto [q_b, t_b] = queue_.front(); 

    // 4. 如果请求时间比插值终点还要晚，说明 q_a 已经没有保留价值了
    // 丢弃 q_a，在下一轮循环中让 q_b 成为新的起点
    // if (t > t_b) {
    //   continue; 
    // }

    // 5. 正常的时间戳线性插值
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    
    if (t < t_a) return q_c;
    
    // 此时 t 一定在 (t_a, t_b] 区间内
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::sb_send(io::sb_VisionToGimbal VisionToGimbal)
{
  sb_tx_data_.mode = VisionToGimbal.mode;
  sb_tx_data_.work_mode = VisionToGimbal.work_mode;
  sb_tx_data_.yaw = VisionToGimbal.yaw;
  sb_tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  sb_tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  sb_tx_data_.pitch = VisionToGimbal.pitch;
  sb_tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  sb_tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
      reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  sb_tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
      reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
      reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  // tools::logger()->info("[Gimbal] mod: {}",tx_data_.mode);
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

// void Gimbal::sb_send(
//   bool control, WorkMode work_mode, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
//   float pitch_acc)
// {
//   tx_data_.mode = control ? (fire ? 2 : 1) : 0;
//   tx_data_.work_mode = static_cast<uint8_t>(work_mode);
//   tx_data_.yaw = yaw;
//   tx_data_.yaw_vel = yaw_vel;
//   tx_data_.yaw_acc = yaw_acc;
//   tx_data_.pitch = pitch;
//   tx_data_.pitch_vel = pitch_vel;
//   tx_data_.pitch_acc = pitch_acc;
//       reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
//   tx_data_.crc16 = tools::get_crc16(
//     reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

//   try {
//     serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
//   } catch (const std::exception & e) {
//     tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
//   }
// }

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  
  // 使用 std::vector 作为动态的滑动窗口缓冲区
  std::vector<uint8_t> rx_buffer;

  while (!quit_) {
    if (error_count > 50000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      rx_buffer.clear(); // 重连时清空遗留数据
      continue;
    }

    // 1. 读取可用数据追加到缓冲区。如果没有数据，阻塞等待至少读 1 字节
    size_t available_bytes = serial_.available();
    size_t bytes_to_read = std::max<size_t>(1, available_bytes); 
    
    try {
      // serial 库重载的 vector read 方法会自动将新数据 append 到 rx_buffer 尾部
      size_t read_len = serial_.read(rx_buffer, bytes_to_read);
      if (read_len == 0) {
        error_count++;
        continue;
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
      error_count++;
      continue;
    }

    // 2. 滑动窗口解析
    // 只要缓冲区里的数据长度足够拼出一帧，就开始解析
    while (rx_buffer.size() >= sizeof(rx_data_)) {
      
      // 检查帧头是否匹配
      if (rx_buffer[0] == 0x5a && rx_buffer[1] == 0x53) {
        
        // 疑似找到一帧数据，拷贝出来进行 CRC 校验
        std::memcpy(&rx_data_, rx_buffer.data(), sizeof(rx_data_));

        if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
          // CRC 校验失败：说明这可能只是载荷里碰巧包含了 0x5a 0x53
          // 窗口向后滑动一个字节，继续寻找真正的帧头
          rx_buffer.erase(rx_buffer.begin());
          error_count++;
          continue; // 仍在 while 循环内，马上进行下一次尝试
        }

        error_count = 0;
        auto t = std::chrono::steady_clock::now();

        // 将这一整帧数据从滑动窗口中彻底移除
        rx_buffer.erase(rx_buffer.begin(), rx_buffer.begin() + sizeof(rx_data_));

        // --- 以下为原有的业务处理逻辑（未修改） ---
        Eigen::Quaterniond q_(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
        auto ypr = tools::eulers(q_, 2, 1, 0);
        
        float yaw = ypr[abs(gimbal_yaw2vision) -  1];
        float pitch = ypr[abs(gimbal_pitch2vision) - 1];
        float roll = ypr[abs(gimbal_roll2vision) - 1];

        yaw = gimbal_yaw2vision > 0 ? yaw : -yaw;
        pitch = gimbal_pitch2vision > 0 ? pitch : -pitch;
        roll = gimbal_roll2vision > 0 ? roll : -roll;

        Eigen::Quaterniond q = 
            Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

        queue_.push({q, t});

        std::lock_guard<std::mutex> lock(mutex_);
        auto ypr_now = tools::eulers(q, 2, 1, 0);
        state_.yaw = ypr_now[0] * 57.3;
        state_.pitch = ypr_now[1] * 57.3;
        
        state_.mode = rx_data_.mode;
        state_.enemy_color = !rx_data_.color;
        state_.bullet_speed = rx_data_.bullet_speed;
        state_.bullet_count = rx_data_.bullet_count;

        switch (rx_data_.mode) {
          case 0: mode_ = GimbalMode::IDLE; break;
          case 1: mode_ = GimbalMode::AUTO_AIM; break;
          case 2: mode_ = GimbalMode::SMALL_BUFF; break;
          case 3: mode_ = GimbalMode::BIG_BUFF; break;
          default:
            mode_ = GimbalMode::IDLE;
            tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
            break;
        }
      } else {
        // 帧头不匹配：弹出最前面的一个字节，窗口向后滑动
        rx_buffer.erase(rx_buffer.begin());
        error_count++;
      }
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io