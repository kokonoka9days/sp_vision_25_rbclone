#include <cmath>
#include <cstdlib>
#include <vector>

#include "big_rune_motion_estimate/LM_IRLS_BigRunePhaseMotionFilter.hpp"
#include "common/power_rune_function.hpp"
#include "json.hpp"

#ifndef AUTO_BUFF_TEST_DEFAULT_CONFIG
#error "AUTO_BUFF_TEST_DEFAULT_CONFIG is required"
#endif

#define CHECK(condition) do { if (!(condition)) std::abort(); } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 2);
  J_POWER_RUNE.initialize(AUTO_BUFF_TEST_DEFAULT_CONFIG, argv[1]);
  LM_IRLS_BigRunePhaseMotionFilter filter;
  filter.set_rotation_direction(RotationDirection::anticlockwise);

  constexpr double A = -0.30;
  constexpr double B = 0.28;
  constexpr double b = 1.18;
  constexpr double omega = 1.94;
  constexpr double C = 0.2;
  const auto start = std::chrono::steady_clock::now();
  bool ready = false;
  for (int i = 0; i < 180; ++i) {
    const double t = i * 0.02;
    double phase = A * std::cos(omega * t) + B * std::sin(omega * t) + b * t + C;
    if (i == 120) phase += 0.8;
    CandidateTarget target;
    target.phase = PRF::normalize_phase<PRF::rad>(phase);
    target.capture_timestamp = start + std::chrono::milliseconds(i * 20);
    target.rune_center = Eigen::Vector3d(0.0, 0.0, 8.0);
    target.armor_module_center = Eigen::Vector3d(0.0, 0.7, 8.0);
    target.start_vector = Eigen::Vector3d::UnitX();
    target.rune_plane_world_normal = Eigen::Vector3d::UnitZ();
    ready = filter.calculate_motion(std::vector<CandidateTarget>{target}) || ready;
  }

  CHECK(ready);
  const auto model = filter.get_motion();
  CHECK(std::isfinite(model.phase_cos_coefficient));
  CHECK(std::isfinite(model.phase_sin_coefficient));
  CHECK(std::isfinite(model.phase_linear_velocity));
  CHECK(model.speed_angular_frequency >= 1.884);
  CHECK(model.speed_angular_frequency <= 2.000);
  return 0;
}
