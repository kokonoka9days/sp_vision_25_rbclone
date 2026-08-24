#pragma once

#include "common/power_rune_global.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

class PowerRuneDiagnostics
{
public:
    static PowerRuneDiagnostics &instance();

    PowerRuneDiagnostics(const PowerRuneDiagnostics &) = delete;
    PowerRuneDiagnostics &operator=(const PowerRuneDiagnostics &) = delete;

    void push_back_predict_phase(const RuneTarget &rune_target,
                                 const double &algorithmic_time,
                                 const double &delay_time,
                                 const double &flying_time);
    void push_back_real_phase(const RuneTimestamp &capture_time,
                              const double &phase);
    void clear();

private:
    PowerRuneDiagnostics() = default;

    struct PredictPhase
    {
        RuneTimestamp reference_time;
        double algorithmic_time;
        double delay_time;
        double flying_time;
        RuneTarget rune_target;
    };

    struct RealPhase
    {
        RuneTimestamp capture_time;
        double real_phase;
    };

    static double calculate_predict_phase_at_time(const PredictPhase &predict_phase,
                                                  const RuneTimestamp &target_time);
    void try_calculate_predict_error();

    std::deque<PredictPhase> m_predict_phase_deque;
    std::mutex m_predict_phase_deque_mutex;
    int m_max_predict_phase_deque_size = 1000;

    std::optional<RealPhase> m_real_phase_opt;
    std::mutex m_real_phase_deque_mutex;
    std::condition_variable m_real_phase_cv;

};
