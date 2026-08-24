#include "function.hpp"

#include <cmath>
#include <ctime>

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

RuneTimestamp function::getNowTimestamp()
{
  return std::chrono::steady_clock::now();
}

double function::timestampMinus(const RuneTimestamp & newer, const RuneTimestamp & older)
{
  return std::chrono::duration<double, std::milli>(newer - older).count();
}

std::string function::getLocalTime()
{
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  if (localtime_r(&now, &local) == nullptr) return {};
  return std::to_string(local.tm_year + 1900) + "_" +
         std::to_string(local.tm_mon + 1) + "_" + std::to_string(local.tm_mday) + "_" +
         std::to_string(local.tm_hour) + "_" + std::to_string(local.tm_min) + "_" +
         std::to_string(local.tm_sec);
}

uint64_t function::to_nanoseconds_since_epoch(const RuneTimestamp & timestamp)
{
  return rune_time::to_nanoseconds(timestamp);
}

double function::calculate_delta_phase(const double & new_phase, const double & old_phase)
{
  constexpr double two_pi = 2.0 * kPi;
  double delta = std::fmod(new_phase - old_phase, two_pi);
  if (delta > kPi) delta -= two_pi;
  if (delta < -kPi) delta += two_pi;
  return delta;
}
