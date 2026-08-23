#ifndef AUTO_AIM__MULTITHREAD__DETECTION_POOL_HPP
#define AUTO_AIM__MULTITHREAD__DETECTION_POOL_HPP

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../model/armor.hpp"
#include "../yolo.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
struct DetectionFrame
{
  int id;
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  std::list<Armor> armors;
};

/** @brief 创建多个独立 YOLO 检测器 @param config_path YAML 配置文件路径 @param number 检测器数量 @param debug 是否启用调试输出 @return 检测器数组 */
inline std::vector<YOLO> create_detectors(
  const std::string & config_path, int number, bool debug)
{
  std::vector<YOLO> detectors;
  detectors.reserve(number);
  for (int i = 0; i < number; ++i) detectors.emplace_back(config_path, debug);
  return detectors;
}

class OrderedDetectionQueue
{
public:
  /** @brief 按帧编号加入检测结果，乱序结果将暂存 @param item 检测帧 */
  void enqueue(const DetectionFrame & item)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (item.id < current_id_) {
      tools::logger()->warn("Discarding stale detection frame {}", item.id);
      return;
    }
    if (item.id != current_id_) {
      buffer_[item.id] = item;
      return;
    }

    main_queue_.push(item);
    ++current_id_;
    auto buffered = buffer_.find(current_id_);
    while (buffered != buffer_.end()) {
      main_queue_.push(std::move(buffered->second));
      buffer_.erase(buffered);
      ++current_id_;
      buffered = buffer_.find(current_id_);
    }
    condition_.notify_one();
  }

  /** @brief 阻塞获取下一帧有序结果 @return 检测帧 */
  DetectionFrame dequeue()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !main_queue_.empty(); });
    auto item = std::move(main_queue_.front());
    main_queue_.pop();
    return item;
  }

  /** @brief 尝试立即获取下一帧结果 @param item 输出检测帧 @return 获取成功时返回 true */
  bool try_dequeue(DetectionFrame & item)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (main_queue_.empty()) return false;
    item = std::move(main_queue_.front());
    main_queue_.pop();
    return true;
  }

  /** @brief 获取主队列和乱序缓冲区的总帧数 @return 待处理帧数 */
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return main_queue_.size() + buffer_.size();
  }

private:
  std::queue<DetectionFrame> main_queue_;
  std::unordered_map<int, DetectionFrame> buffer_;
  int current_id_ = 1;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__MULTITHREAD__DETECTION_POOL_HPP
