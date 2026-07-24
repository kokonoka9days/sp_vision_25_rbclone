#ifndef PERIODIC_MOTION_ANALYZER_HPP 
#define PERIODIC_MOTION_ANALYZER_HPP

#include <iostream>
#include <cmath>
#include <chrono>
using std::isnan;
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

#include "math_tools.hpp"

namespace tools
{

class FFTExample {
public:
    using Buffer = boost::circular_buffer<double>;


    FFTExample(size_t max_points = 400)
        : t_buf_(max_points), val_buf_(max_points) {}

    void add_sample(double t, double val) {
        t_buf_.push_back(t);
        val_buf_.push_back(val);
    }

    // 触发分析，返回是否检测到周期运动
    bool analyze(bool force = false) ;

    // 获取加速度函数（在当前时间 t 的值）
    double get_acceleration(double t) const {
        if (!is_periodic_) return 0.0;
        double omega = 2 * M_PI * last_freq_;
        return -omega * omega * last_amp_ * std::cos(omega * t + last_phase_ + phase_offset_);
    }

    // 获取原始拟合函数值
    double get_value(double t) const {
        if (!is_periodic_) return 0.0;
        return last_amp_ * std::cos(2 * M_PI * last_freq_ * t + last_phase_ + phase_offset_); 
    }

    double get_val_buf_front() const { return val_buf_.front(); }

    bool get_is_periodic() const { return is_periodic_; }
    void set_phase_offset(double offset) { phase_offset_ = offset; }

private:
    Buffer t_buf_, val_buf_;
    bool is_periodic_ = false; //是否做周期运动
    double last_freq_ = 0.0, last_amp_ = 0.0, last_phase_ = 0.0, last_period_ = 0.0;
    double phase_offset_ = 0.0; //相位偏移
    size_t min_points_ = 400;   //储存400个数据
    size_t analysis_interval_frames_ = 30;  // 每30帧分析一次
    size_t last_analysis_frames_ = 0;

    double detect_period_by_autocorr(const Eigen::VectorXd& x, double dt) ;

    bool extract_base_harmonic(const Eigen::VectorXd& x, double dt,
                               double& freq, double& amp, double& phase) ;
};    


} // namespace tools



#endif
