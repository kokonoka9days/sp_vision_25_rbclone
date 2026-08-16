#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "tools/thread_pool.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main()
{
  bool rejected_empty_pool = false;
  try {
    tools::ThreadPool invalid(0);
  } catch (const std::invalid_argument &) {
    rejected_empty_pool = true;
  }
  CHECK(rejected_empty_pool);

  std::atomic_int completed{0};
  {
    tools::ThreadPool pool(2);
    for (int i = 0; i < 12; ++i) {
      pool.enqueue([&completed] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++completed;
      });
    }
  }
  CHECK(completed.load() == 12);
  return 0;
}
