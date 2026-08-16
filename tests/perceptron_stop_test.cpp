#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include "tasks/omniperception/perceptron.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

namespace
{
class IdleFrameSource final : public omniperception::IFrameSource
{
public:
  bool read_for(
    cv::Mat &, std::chrono::steady_clock::time_point &,
    std::chrono::milliseconds timeout) override
  {
    std::this_thread::sleep_for(std::min(timeout, std::chrono::milliseconds(5)));
    return false;
  }

  const std::string & name() const override { return name_; }

private:
  std::string name_ = "offline";
};

class EmptyDetector final : public auto_aim::IArmorDetector
{
public:
  std::list<auto_aim::Armor> detect(const cv::Mat &, int) override { return {}; }
};
}  // namespace

int main(int argc, char ** argv)
{
  CHECK(argc == 2);
  std::vector<std::shared_ptr<omniperception::IFrameSource>> sources{
    std::make_shared<IdleFrameSource>(), std::make_shared<IdleFrameSource>()};
  std::vector<std::shared_ptr<auto_aim::IArmorDetector>> detectors{
    std::make_shared<EmptyDetector>(), std::make_shared<EmptyDetector>()};

  const auto start = std::chrono::steady_clock::now();
  {
    omniperception::Perceptron perceptron(sources, detectors, argv[1]);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));
  return 0;
}
