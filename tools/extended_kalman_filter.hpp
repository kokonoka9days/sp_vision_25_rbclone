#ifndef TOOLS__EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

namespace tools
{
class ExtendedKalmanFilter
{
public:
  Eigen::VectorXd x;
  Eigen::MatrixXd P;

  /** @brief 默认构造未初始化的扩展卡尔曼滤波器 */
  ExtendedKalmanFilter() = default;

  /**
   * @brief 使用初始状态和协方差构造扩展卡尔曼滤波器
   * @param x0 初始状态向量
   * @param P0 初始协方差矩阵
   * @param x_add 状态加法函数，可用于处理角度等非线性状态
   * @throws std::invalid_argument 如果状态与协方差维度不匹配
   */
  ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  /**
   * @brief 使用线性状态转移模型执行预测
   * @param F 状态转移矩阵
   * @param Q 过程噪声协方差矩阵
   * @return 预测后的状态向量
   */
  Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  /**
   * @brief 使用非线性状态转移模型执行预测
   * @param F 状态转移函数在当前状态处的雅可比矩阵
   * @param Q 过程噪声协方差矩阵
   * @param f 非线性状态转移函数
   * @return 预测后的状态向量
   */
  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  /**
   * @brief 使用线性观测模型执行更新
   * @param z 观测向量
   * @param H 观测矩阵
   * @param R 观测噪声协方差矩阵
   * @param z_subtract 观测残差计算函数
   * @return 更新后的状态向量
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  /**
   * @brief 使用非线性观测模型执行更新
   * @param z 观测向量
   * @param H 观测函数在当前状态处的雅可比矩阵
   * @param R 观测噪声协方差矩阵
   * @param h 将状态映射到观测空间的函数
   * @param z_subtract 观测残差计算函数
   * @return 更新后的状态向量
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  /** @brief 获取最近一次卡方检验数据 @return 检验指标名称与数值的映射 */
  std::map<std::string, double> get_Chi_Square_test_datas() const{
    return data;
  };
  std::map<std::string, double> data;  //卡方检验数据
  std::deque<int> recent_nis_failures;
  size_t window_size = 100;
  double last_nis;

private:
  Eigen::MatrixXd I;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

  int nees_count_ = 0;
  int nis_count_ = 0;
  int total_count_ = 0;
};

}  // namespace tools

#endif  // TOOLS__EXTENDED_KALMAN_FILTER_HPP
