#include "fft.hpp"

#include <boost/math/interpolators/pchip.hpp>
#include <unsupported/Eigen/FFT>
#include <numeric>

#include "logger.hpp"

namespace tools
{

bool FFTExample::analyze(bool force) {
    // 加锁拷贝缓冲区数据到局部变量
    std::vector<double> t_vec, val_vec;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!force && (t_buf_.size() < min_points_)) {
            return is_periodic_;
        }
        last_analysis_frames_ = 0;
        if (t_buf_.size() < min_points_) return false;

        t_vec.assign(t_buf_.begin(), t_buf_.end());
        val_vec.assign(val_buf_.begin(), val_buf_.end());
    } // 锁在此释放，后续计算不再持有锁

    this->mean_val = std::accumulate(val_vec.begin(), val_vec.end(), 0.0) / val_vec.size();

    double t_min = t_vec.front(), t_max = t_vec.back();
    int n_uniform = static_cast<int>((t_max - t_min) / 0.01) + 1;
    std::vector<double> t_uniform(n_uniform);
    for (int i = 0; i < n_uniform; ++i)
        t_uniform[i] = t_min + i * 0.01;

    auto pchip = boost::math::interpolators::pchip(
        std::move(t_vec), std::move(val_vec));
    Eigen::VectorXd val_uniform(n_uniform);
    for (int i = 0; i < n_uniform; ++i)
        val_uniform(i) = pchip(t_uniform[i]);

    double period = detect_period_by_autocorr(val_uniform, 0.01);
    if (period <= 0.0) {
        is_periodic_ = false;
        return false;
    }

    double freq, amp, phase;
    if (!extract_base_harmonic(val_uniform, 0.01, freq, amp, phase)) {
        is_periodic_ = false;
        return false;
    }

    is_periodic_ = true;
    last_freq_ = freq;
    last_amp_ = amp;
    last_phase_ = phase;
    last_period_ = 1.0 / freq;

    return true;
}

double FFTExample::detect_period_by_autocorr(const Eigen::VectorXd& x, double dt) {
    Eigen::VectorXd xm = x.array() - x.mean();
    int N = x.size();
    int max_lag = N / 2;
    double max_corr = 0.0;
    int best_lag = -1;
    for (int lag = 1; lag < max_lag; ++lag) {
        double corr = xm.head(N - lag).dot(xm.tail(N - lag));
        if (corr > max_corr) {
            max_corr = corr;
            best_lag = lag;
        }
    }
    double norm = xm.squaredNorm();
    if (norm < 1e-6) {
        tools::logger()->warn("[FFT] 信号能量过低 norm = {}, 无法检测周期", norm);
        return -1.0;
    }
    double peak = max_corr / norm;
    if (peak < 0.5) {
        tools::logger()->warn("[FFT] 信号周期性不明显peak = {}, 无法检测周期", peak);
        return -1.0;
    }
    return best_lag * dt;
}

bool FFTExample::extract_base_harmonic(const Eigen::VectorXd& x, double dt,
                            double& freq, double& amp, double& phase) {
    int N = x.size();
    Eigen::FFT<double> fft;
    Eigen::VectorXcd X(N);
    fft.fwd(X, x);
    int n2 = N / 2;
    double max_mag = 0.0;
    int max_idx = 0;
    for (int i = 1; i <= n2; ++i) {
        double mag = std::abs(X(i));
        if (mag > max_mag) {
            max_mag = mag;
            max_idx = i;
        }
    }
    if (max_idx == 0) return false;
    freq = max_idx / (N * dt);
    amp = 2.0 * max_mag / N;
    phase = std::arg(X(max_idx));
    return true;
}

} // namespace tools