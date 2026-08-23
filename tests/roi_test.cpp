#include <cstdlib>
#include <stdexcept>

#include "tasks/auto_aim/detection/roi.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main()
{
  const auto full_remainder = auto_aim::resolve_roi({100, 50, -1, -1}, {640, 360});
  CHECK(full_remainder == cv::Rect(100, 50, 540, 310));

  const auto explicit_roi = auto_aim::resolve_roi({10, 20, 100, 80}, {640, 360});
  CHECK(explicit_roi == cv::Rect(10, 20, 100, 80));

  bool rejected = false;
  try {
    static_cast<void>(auto_aim::resolve_roi({600, 20, 100, 80}, {640, 360}));
  } catch (const std::out_of_range &) {
    rejected = true;
  }
  CHECK(rejected);
  return 0;
}
