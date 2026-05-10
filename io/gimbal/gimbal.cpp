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
    serial_.setBaudrate(115200);
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
    auto [q_a, t_a] = queue_.pop();

    if (queue_.empty()) {
      return q_a;
    }

    auto [q_b, t_b] = queue_.front();

    if (t > t_b) {
      continue;
    }

    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    
    // 【修复1】：防止 t_ab 过小导致的除零错或 K 值爆炸
    if (t_ab < 1e-4) { 
      return q_a; // 两次数据间隔小于 0.1ms，直接返回，不插值
    }

    auto k = t_ac / t_ab;
    
    // 【修复2】：限制 k 的范围。
    // 允许合理的稍微外推（比如 -1.0 到 2.0 之间），但是拒绝离谱的疯狂外推
    k = std::clamp(k, -1.0, 2.0);

    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();

    // 如果目标时间比我们能查到的最老时间还老，就只能认命返回推算值了
    if (t < t_a) return q_c;

    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

Eigen::Vector3d Gimbal::ypr(std::chrono::steady_clock::time_point t)
{
  return tools::eulers(q(t), 2, 1, 0);
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
  // sb_tx_data_.crc16 = tools::get_crc16(
    // reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  // tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  // tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  // tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  // tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  // tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
      // reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  // tx_data_.crc16 = tools::get_crc16(
  //   reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

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
  // tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  // tx_data_.yaw_vel = yaw_vel;
  // tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  // tx_data_.pitch_vel = pitch_vel;
  // tx_data_.pitch_acc = pitch_acc;
      // reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  // tx_data_.crc16 = tools::get_crc16(
  //   reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::drone_send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  // tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  drone_tx_date.yaw = yaw;
  // tx_data_.yaw_vel = yaw_vel;
  // tx_data_.yaw_acc = yaw_acc;
  drone_tx_date.pitch = pitch;
  // tx_data_.pitch_vel = pitch_vel;
  // tx_data_.pitch_acc = pitch_acc;
      // reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) ;
  drone_tx_date.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&drone_tx_date), sizeof(drone_tx_date) - sizeof(drone_tx_date.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&drone_tx_date), sizeof(drone_tx_date));
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

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  int frame_count = 0;
  auto last_print_time = std::chrono::steady_clock::now();
  std::vector<uint8_t> rx_buffer;
  constexpr size_t FRAME_SIZE = sizeof(GimbalToVision);

  while (!quit_) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 1) {
      tools::logger()->info("[Gimbal] 每秒成功接收帧数: {}", frame_count);
      frame_count = 0;
      last_print_time = now;
    }
    if (error_count > 50000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      rx_buffer.clear();
      reconnect();
      continue;
    }

    // 1. 把串口当前所有可读数据追加到本地缓冲区
    bool read_something = false;
    try {
      size_t avail = serial_.available();
      if (avail > 0) {
        std::vector<uint8_t> new_data;
        serial_.read(new_data, avail);
        if (!new_data.empty()) {
          rx_buffer.insert(rx_buffer.end(), new_data.begin(), new_data.end());
          read_something = true;
        }
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
      error_count++;
      continue;
    }

    // 2. 在缓冲区中逐字节查找帧头 0x5a，凑够完整一帧后校验 CRC
    bool found_frame = false;
    for (size_t i = 0; i + FRAME_SIZE <= rx_buffer.size(); ++i) {
      if (rx_buffer[i] == 0x5a) {
        std::memcpy(&rx_data_, &rx_buffer[i], FRAME_SIZE);

        if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
          // 这个 0x5a 只是数据里的巧合，继续往后找
          continue;
        }

        // 找到有效帧，移除本帧及之前的所有数据
        rx_buffer.erase(rx_buffer.begin(), rx_buffer.begin() + i + FRAME_SIZE);
        found_frame = true;
        break;
      }
    }

    if (!found_frame) {
      // 防止缓冲区无限增长：没找齐帧时只保留末尾可能属于下一帧的数据
      if (rx_buffer.size() > FRAME_SIZE * 4) {
        rx_buffer.erase(rx_buffer.begin(), rx_buffer.end() - FRAME_SIZE + 1);
      }
      // 没读到新数据时短暂 sleep，避免空转占满 CPU
      if (!read_something) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      continue;
    }

    // 3. 记录成功接收到有效帧的时间戳
    auto t = std::chrono::steady_clock::now();

    // --- 以下为原本的数据处理逻辑，保持不变 ---

    error_count = 0;
    frame_count++;

    // 电控直接发送欧拉角，先做坐标系映射
    Eigen::Vector3d ypr_raw(rx_data_.yaw, rx_data_.pitch, rx_data_.roll);

    float yaw = ypr_raw[abs(gimbal_yaw2vision) -  1];
    float pitch = ypr_raw[abs(gimbal_pitch2vision) - 1];
    float roll = ypr_raw[abs(gimbal_roll2vision) - 1];

    yaw = gimbal_yaw2vision > 0 ? yaw : -yaw;
    pitch = gimbal_pitch2vision > 0 ? pitch : -pitch;
    roll = gimbal_roll2vision > 0 ? roll : -roll;

    // 电控发来的是角度制，先转成弧度再构建四元数入队（保证插值平滑）
    constexpr double D2R = M_PI / 180.0;
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw * D2R, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch * D2R, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll * D2R, Eigen::Vector3d::UnitX());
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);
    state_.yaw = yaw;
    state_.pitch = pitch;
    state_.roll = roll;

  //   state_.mode = rx_data_.mode;
  //   state_.enemy_color = !rx_data_.color;
  //   state_.bullet_speed = rx_data_.bullet_speed;
  //   state_.bullet_count = rx_data_.bullet_count;
  //   // rx_data_.mode = 2;
  //   state_.mode = rx_data_.mode;

  //   switch (rx_data_.mode) {
  //     case 0:
  //       mode_ = GimbalMode::IDLE;
  //       break;
  //     case 1:
  //       mode_ = GimbalMode::AUTO_AIM;
  //       break;
  //     case 2:
  //       mode_ = GimbalMode::SMALL_BUFF;
  //       break;
  //     case 3:
  //       mode_ = GimbalMode::BIG_BUFF;
  //       break;
  //     default:
  //       mode_ = GimbalMode::IDLE;
  //       tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
  //       break;
  //   }
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