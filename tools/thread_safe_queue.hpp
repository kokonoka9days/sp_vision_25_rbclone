#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace tools
{
enum class OverflowPolicy
{
  DropNewest,
  DropOldest,
  Block
};

enum class QueuePushResult
{
  Pushed,
  DroppedNewest,
  DroppedOldest,
  Closed
};

template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  /** @brief 构造有界线程安全队列 @param max_size 最大元素数 @param full_handler 队列溢出时的回调 @throws std::invalid_argument 当 max_size 为 0 */
  explicit ThreadSafeQueue(
    size_t max_size, std::function<void(void)> full_handler = [] {})
  : ThreadSafeQueue(
      max_size, PopWhenFull ? OverflowPolicy::DropOldest : OverflowPolicy::DropNewest,
      std::move(full_handler))
  {
  }

  /** @brief 构造具有指定溢出策略的队列 @param max_size 最大元素数 @param overflow_policy 溢出策略 @param full_handler 队列溢出时的回调 @throws std::invalid_argument 当 max_size 为 0 */
  ThreadSafeQueue(
    size_t max_size, OverflowPolicy overflow_policy,
    std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), overflow_policy_(overflow_policy), full_handler_(std::move(full_handler))
  {
    if (max_size_ == 0) throw std::invalid_argument("ThreadSafeQueue max_size must be positive");
  }

  /** @brief 复制元素并压入队列 @param value 元素 @return 入队结果 */
  QueuePushResult push(const T & value) { return push_impl(T(value)); }
  /** @brief 移动元素并压入队列 @param value 元素 @return 入队结果 */
  QueuePushResult push(T && value) { return push_impl(std::move(value)); }

  /** @brief 阻塞等待并弹出元素 @param value 输出元素；队列关闭且为空时保持不变 */
  void pop(T & value)
  {
    auto result = wait_pop();
    if (result.has_value()) value = std::move(*result);
  }

  /** @brief 尝试立即弹出元素 @param value 输出元素 @return 成功弹出时返回 true */
  bool try_pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    value = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_condition_.notify_one();
    return true;
  }

  /** @brief 阻塞等待并返回元素 @return 弹出的元素 @throws std::runtime_error 当队列关闭且为空 */
  T pop()
  {
    auto result = wait_pop();
    if (!result.has_value()) throw std::runtime_error("pop on a closed ThreadSafeQueue");
    return std::move(*result);
  }

  /** @brief 阻塞等待元素或队列关闭 @return 弹出的元素；队列关闭且为空时返回空值 */
  std::optional<T> wait_pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    return pop_locked(lock);
  }

  /** @brief 在限定时间内等待元素 @tparam Rep 时长数值类型 @tparam Period 时长周期类型 @param timeout 最长等待时间 @return 弹出的元素；超时或关闭时返回空值 */
  template <typename Rep, typename Period>
  std::optional<T> wait_pop_for(const std::chrono::duration<Rep, Period> & timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_condition_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); })) {
      return std::nullopt;
    }
    return pop_locked(lock);
  }

  /** @brief 阻塞获取队首元素的副本 @return 队首元素 @throws std::runtime_error 当队列关闭且为空 */
  T front()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) throw std::runtime_error("front on a closed ThreadSafeQueue");
    return queue_.front();
  }

  /** @brief 获取队尾元素的副本 @param value 输出元素 @return 队列非空时返回 true */
  bool back(T & value)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    value = queue_.back();
    return true;
  }

  /** @brief 判断队列是否为空 @return 为空时返回 true */
  bool empty() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /** @brief 获取队列元素数 @return 当前元素数 */
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /** @brief 清空队列并唤醒等待入队的线程 */
  void clear()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      while (!queue_.empty()) queue_.pop();
    }
    not_full_condition_.notify_all();
  }

  /** @brief 关闭队列并唤醒全部等待线程 */
  void close()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_condition_.notify_all();
    not_full_condition_.notify_all();
  }

  /** @brief 查询队列是否关闭 @return 已关闭时返回 true */
  bool closed() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

private:
  /** @brief 按溢出策略压入元素 @param value 待入队元素 @return 入队结果 */
  QueuePushResult push_impl(T value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (overflow_policy_ == OverflowPolicy::Block) {
      not_full_condition_.wait(lock, [this] { return closed_ || queue_.size() < max_size_; });
    }
    if (closed_) return QueuePushResult::Closed;

    QueuePushResult result = QueuePushResult::Pushed;
    if (queue_.size() >= max_size_) {
      if (overflow_policy_ == OverflowPolicy::DropNewest) {
        lock.unlock();
        full_handler_();
        return QueuePushResult::DroppedNewest;
      }
      queue_.pop();
      result = QueuePushResult::DroppedOldest;
    }

    queue_.push(std::move(value));
    lock.unlock();
    not_empty_condition_.notify_one();
    if (result == QueuePushResult::DroppedOldest) full_handler_();
    return result;
  }

  /** @brief 在已持锁状态下弹出队首元素 @param lock 当前互斥锁 @return 弹出的元素；队列为空时返回空值 */
  std::optional<T> pop_locked(std::unique_lock<std::mutex> & lock)
  {
    if (queue_.empty()) return std::nullopt;
    T value = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_condition_.notify_one();
    return value;
  }

  std::queue<T> queue_;
  size_t max_size_;
  OverflowPolicy overflow_policy_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::condition_variable not_full_condition_;
  std::function<void(void)> full_handler_;
  bool closed_ = false;
};

}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP
