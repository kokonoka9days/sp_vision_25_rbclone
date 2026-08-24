#pragma once

#include "runtime_types.hpp"

#include <cmath>
#include <cstdint>

inline constexpr double kPowerRunePi = 3.14159265358979323846;

namespace power_rune_function
{

struct rad
{
    static constexpr double period = 2.0 * kPowerRunePi;
};
struct deg
{
    static constexpr double period = 360.0;
};

template<typename T>
double calculate_delta_phase(double new_phase, double old_phase)
{
    constexpr double period = T::period;
    constexpr double half_period = period * 0.5;

    double delta_phase = new_phase - old_phase;
    while (delta_phase > half_period)
    {
        delta_phase -= period;
    }
    while (delta_phase < -half_period)
    {
        delta_phase += period;
    }
    return delta_phase;
}

template<typename T>
double normalize_phase(double phase)
{
    constexpr double period = T::period;
    constexpr double half_period = period * 0.5;

    phase = std::fmod(phase + half_period, period);

    if (phase < 0)
    {
        phase += period;
    }

    return phase - half_period;
}

inline RuneTimestamp timestamp_from_nanoseconds(uint64_t timestamp_ns)
{
    return rune_time::from_nanoseconds(timestamp_ns);
}

} // namespace power_rune_function
namespace PRF = power_rune_function;
