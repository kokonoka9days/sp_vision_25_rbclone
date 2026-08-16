#pragma once

#include <Eigen/Dense>
#include <deque>
#include <iostream>
#include <limits>
#include <vector>

namespace tools
{

class RansacSineFitter
{
public:
  struct Result
  {
    double A = 0.0;
    double omega = 0.0;
    double phi = 0.0;
    double C = 0.0;
    int inliers = 0;
    double rms = std::numeric_limits<double>::infinity();
  };
  Result best_result_;

  /** @brief 构造 RANSAC 正弦拟合器 @param max_iterations 最大迭代次数 @param threshold 内点残差阈值 @param min_omega 最小角频率 @param max_omega 最大角频率 */
  RansacSineFitter(int max_iterations, double threshold, double min_omega, double max_omega);

  /** @brief 添加拟合样本 @param t 时间 @param v 观测值 */
  void add_data(double t, double v);

  /** @brief 对当前样本执行正弦模型拟合并更新 best_result_ */
  void fit();

  /** @brief 清空样本和拟合结果 */
  void clear();

  /** @brief 获取当前样本数 @return 样本数量 */
  size_t sample_count() const;

  /** @brief 获取样本覆盖的时间跨度 @return 时间跨度 */
  double time_span() const;

  /** @brief 获取最佳模型的内点比例 @return 范围为 0 到 1 的内点比例 */
  double inlier_ratio() const;

  /** @brief 判断当前拟合是否可用 @param min_samples 最小样本数 @param min_time_span 最小时间跨度 @param min_inlier_ratio 最小内点比例 @param max_rms 最大均方根误差 @return 满足全部条件时返回 true */
  bool ready(
    size_t min_samples, double min_time_span, double min_inlier_ratio,
    double max_rms = std::numeric_limits<double>::infinity()) const;

  /** @brief 计算正弦模型值 @param t 时间 @param A 振幅 @param omega 角频率 @param phi 相位 @param C 直流偏置 @return 模型在 t 时刻的值 */
  double sine_function(double t, double A, double omega, double phi, double C)
  {
    return A * std::sin(omega * t + phi) + C;
  }

private:
  double threshold_;
  double min_omega_;
  double max_omega_;
  std::deque<std::pair<double, double>> fit_data_;

  /** @brief 在固定角频率下执行加权线性拟合 @param omega 角频率 @param weights 样本权重 @param params 输出的线性参数 @return 拟合成功时返回 true */
  bool fit_weighted_model(
    double omega, const std::vector<double> & weights, Eigen::Vector3d & params) const;

  /** @brief 评估一组正弦模型参数 @param A 振幅 @param omega 角频率 @param phi 相位 @param C 直流偏置 @return 包含内点数和误差的评估结果 */
  Result evaluate_model(double A, double omega, double phi, double C) const;
};

}  // namespace tools
