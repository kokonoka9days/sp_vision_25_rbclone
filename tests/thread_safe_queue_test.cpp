#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <thread>

#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

#define CHECK(condition)          \
  do {                            \
    if (!(condition)) std::abort(); \
  } while (false)

int main()
{
  using tools::OverflowPolicy;
  using tools::QueuePushResult;
  using tools::ThreadSafeQueue;

  {
    ThreadSafeQueue<int> queue(1, OverflowPolicy::DropNewest);
    CHECK(queue.push(1) == QueuePushResult::Pushed);
    CHECK(queue.push(2) == QueuePushResult::DroppedNewest);
    CHECK(queue.pop() == 1);
  }

  {
    ThreadSafeQueue<int> queue(1, OverflowPolicy::DropOldest);
    CHECK(queue.push(1) == QueuePushResult::Pushed);
    CHECK(queue.push(2) == QueuePushResult::DroppedOldest);
    CHECK(queue.pop() == 2);
  }

  {
    ThreadSafeQueue<int> queue(1, OverflowPolicy::Block);
    CHECK(queue.push(1) == QueuePushResult::Pushed);

    auto blocked_push = std::async(std::launch::async, [&queue] { return queue.push(2); });
    CHECK(blocked_push.wait_for(20ms) == std::future_status::timeout);
    CHECK(queue.pop() == 1);
    CHECK(blocked_push.wait_for(200ms) == std::future_status::ready);
    CHECK(blocked_push.get() == QueuePushResult::Pushed);
    CHECK(queue.pop() == 2);
  }

  {
    ThreadSafeQueue<int> queue(1);
    auto waiting_pop = std::async(std::launch::async, [&queue] { return queue.wait_pop(); });
    CHECK(waiting_pop.wait_for(20ms) == std::future_status::timeout);
    queue.close();
    CHECK(waiting_pop.wait_for(200ms) == std::future_status::ready);
    CHECK(!waiting_pop.get().has_value());
    CHECK(queue.push(1) == QueuePushResult::Closed);
  }

  {
    ThreadSafeQueue<int> queue(1, OverflowPolicy::Block);
    CHECK(queue.push(1) == QueuePushResult::Pushed);
    auto blocked_push = std::async(std::launch::async, [&queue] { return queue.push(2); });
    CHECK(blocked_push.wait_for(20ms) == std::future_status::timeout);
    queue.close();
    CHECK(blocked_push.wait_for(200ms) == std::future_status::ready);
    CHECK(blocked_push.get() == QueuePushResult::Closed);
    CHECK(queue.pop() == 1);
    CHECK(!queue.wait_pop_for(1ms).has_value());
  }

  {
    ThreadSafeQueue<int> queue(1);
    const auto start = std::chrono::steady_clock::now();
    CHECK(!queue.wait_pop_for(20ms).has_value());
    CHECK(std::chrono::steady_clock::now() - start >= 15ms);
  }

  return 0;
}
