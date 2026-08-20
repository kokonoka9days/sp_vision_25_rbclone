#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "tools/yaw_delay_model.hpp"

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
      std::abort();                                                             \
    }                                                                           \
  } while (false)

int main()
{
  using Clock = std::chrono::steady_clock;
  const std::vector<tools::YawDelayPoint> positive{{0.0, 0.002}, {3.0, 0.004}, {6.0, 0.008}, {10.0, 0.014}};
  const std::vector<tools::YawDelayPoint> negative{{0.0, 0.002}, {3.0, 0.005}, {6.0, 0.009}, {10.0, 0.016}};
  tools::YawDelayModel model(positive, negative, 0.002, 0.2, 0.05);
  const auto t0 = Clock::now();

  CHECK(model.enabled());
  CHECK(std::abs(model.query(4.5, 4.5, t0) - 0.006) < 1e-9);
  model.reset();
  CHECK(std::abs(model.query(-7.5, -7.5, t0) - 0.011625) < 1e-9);
  model.reset();
  CHECK(std::abs(model.query(20.0, 20.0, t0) - 0.014) < 1e-9);
  model.reset();
  CHECK(std::abs(model.query(-20.0, -20.0, t0) - 0.016) < 1e-9);

  model.reset();
  CHECK(std::abs(model.query(3.0, 3.0, t0) - 0.004) < 1e-9);
  CHECK(std::abs(model.query(0.1, 0.1, t0) - (0.002 + 0.1 / 3.0 * 0.002)) < 1e-9);
  CHECK(std::abs(model.query(-3.0, 3.0, t0) - 0.007) < 1e-9);
  CHECK(model.reversal_active(t0));
  CHECK(std::abs(model.query(-3.0, -3.0, t0 + std::chrono::milliseconds(20)) - 0.007) < 1e-9);
  CHECK(std::abs(model.query(-3.0, -3.0, t0 + std::chrono::milliseconds(60)) - 0.005) < 1e-9);
  CHECK(model.direction(0.1) == 0);

  const auto expect_invalid = [](const std::vector<tools::YawDelayPoint> & curve) {
    bool thrown = false;
    try {
      tools::YawDelayModel(curve, curve, 0.0, 0.2, 0.05);
    } catch (const std::invalid_argument &) {
      thrown = true;
    }
    CHECK(thrown);
  };
  expect_invalid({{0.0, 0.002}, {0.0, 0.003}});
  expect_invalid({{0.0, -0.001}});
  expect_invalid({{0.0, 0.201}});

  std::vector<tools::YawDelaySample> samples;
  constexpr double sample_period = 0.005;
  constexpr double known_delay = 0.05;
  for (int i = 0; i <= 500; ++i) {
    const double time = i * sample_period;
    const auto signal = [](double t) {
      return 0.4 * std::sin(2.0 * M_PI * t) + 0.1 * std::sin(5.0 * M_PI * t);
    };
    samples.push_back({time, signal(time), signal(time - known_delay)});
  }
  const auto estimate = tools::estimate_yaw_delay(samples);
  CHECK(estimate.valid);
  CHECK(std::abs(estimate.delay_s - known_delay) <= 0.007);
  CHECK(estimate.correlation > 0.95);
  return 0;
}
