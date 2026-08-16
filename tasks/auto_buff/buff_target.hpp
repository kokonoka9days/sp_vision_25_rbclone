#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <deque>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>
#include <memory>

#include "buff_detector.hpp"
#include "buff_config.hpp"
#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/ransac_sine_fitter.hpp"

namespace auto_buff
{
class PhaseDirectionTracker
{
public:
  /** @brief 构造相位方向跟踪器 @param confirm_intervals 确认旋转方向所需连续间隔数 */
  explicit PhaseDirectionTracker(int confirm_intervals = 3)
  : confirm_intervals_(confirm_intervals) {}
  /** @brief 重设相位参考 @param phase 新相位 @param preserve_direction 是否保留已确认方向 */
  void rebase(double phase, bool preserve_direction);
  /** @brief 平移内部相位参考 @param delta 相位增量，单位 rad */
  void shift_reference(double delta);
  /** @brief 加入新相位并更新方向投票 @param phase 观测相位，单位 rad */
  void update(double phase);
  /** @brief 清空方向和历史样本 */
  void reset();
  /** @brief 获取旋转方向 @return 顺时针/逆时针符号，未确认时为 0 */
  int direction() const { return direction_; }
  /** @brief 查询方向是否确认 @return 已确认时返回 true */
  bool ready() const { return direction_ != 0; }

private:
  int confirm_intervals_ = 3;
  bool has_last_phase_ = false;
  double last_phase_ = 0.0;
  int direction_ = 0;
  int score_ = 0;
  int reverse_candidate_direction_ = 0;
  int reverse_confirm_count_ = 0;
  std::deque<double> deltas_;
  std::deque<int> votes_;
};

/// Target 基类

class Target
{
public:
  /** @brief 使用默认配置构造目标 */
  Target();
  /** @brief 使用指定配置构造目标 @param config 能量机关配置 */
  explicit Target(BuffConfig config);
  /** @brief 销毁目标基类 */
  virtual ~Target() = default;
  /** @brief 使用可选观测更新目标 @param p 三维能量机关观测 @param timestamp 观测时间戳 */
  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;  // 纯虚函数

  /** @brief 从多轨迹观测中选择并更新目标 @param observations 三维观测列表 @param timestamp 观测时间戳 */
  void get_target(
    const std::vector<PowerRune> & observations,
    std::chrono::steady_clock::time_point & timestamp);

  /** @brief 将目标状态向前预测 @param dt 预测时长，单位 s */
  virtual void predict(double dt) = 0;  // 纯虚函数

  /** @brief 将能量机关局部点转换为世界坐标 @param point_in_buff 局部坐标 @return 世界坐标 */
  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  /** @brief 获取能量机关到世界坐标系旋转 @return 旋转矩阵 */
  virtual Eigen::Matrix3d rotation_buff2world() const;

  /** @brief 查询目标是否无有效解 @return 无法求解时返回 true */
  bool is_unsolve() const;

  /** @brief 查询状态是否足以控制云台 @return 可控制时返回 true */
  bool can_control() const;

  /** @brief 查询状态是否足以执行运动预测 @return 可预测时返回 true */
  bool prediction_ready() const;

  /** @brief 获取目标就绪状态 @return 就绪状态枚举 */
  TargetReadiness readiness() const { return readiness_; }

  /** @brief 查询当前是否处于盲跟踪 @return 无新观测而在外推时返回 true */
  bool is_blind() const;

  /** @brief 判断当前状态是否允许开火 @param now 当前时间 @return 允许开火时返回 true */
  bool can_fire(std::chrono::steady_clock::time_point now) const;

  /** @brief 获取累计重置次数 @return 重置次数 */
  int reset_count() const { return reset_count_; }

  /** @brief 获取目标配置 @return 配置只读引用 */
  const BuffConfig & config() const { return config_; }

  /** @brief 重置滤波、时序和可用性状态 */
  virtual void reset();

  /** @brief 获取 EKF 状态副本 @return 状态向量 */
  Eigen::VectorXd ekf_x() const;

  double spd = 0;  //调试用

  /** @brief 克隆具体目标对象 @return 独立目标副本 */
  virtual std::unique_ptr<Target> clone() const = 0;

protected:
  const BuffConfig config_;
  /** @brief 使用首次观测初始化目标 @param nowtime 相对时间，单位 s @param p 三维观测 */
  virtual void init(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

  /** @brief 使用新观测更新目标 @param nowtime 相对时间，单位 s @param p 三维观测 */
  virtual void update(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

  /** @brief 将绝对时间戳转换为目标相对时间 @param timestamp 时间戳 @return 相对秒数 */
  double relative_time(std::chrono::steady_clock::time_point timestamp);

  /** @brief 在无测量时执行有限时间外推 @param timestamp 当前时间戳 @return 仍可保留目标时返回 true */
  bool predict_without_measurement(std::chrono::steady_clock::time_point timestamp);

  /** @brief 记录最新测量及其质量 @param p 三维观测 @param timestamp 时间戳 */
  void record_measurement(
    const PowerRune & p, std::chrono::steady_clock::time_point timestamp);

  /** @brief 更新能量机关平面基向量 @param p 三维观测 @param initialize 是否初始化基向量 @return 相位参考修正量 */
  double update_plane_basis(const PowerRune & p, bool initialize);

  /** @brief 测量扇叶相位并展开到参考相位附近 @param p 三维观测 @param reference 参考相位 @return 连续相位 */
  double measure_phase(const PowerRune & p, double reference) const;

  Eigen::VectorXd x0_;
  Eigen::MatrixXd P0_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd H_;
  Eigen::MatrixXd R_;
  tools::ExtendedKalmanFilter ekf_;
  double lasttime_ = 0;
  bool first_in_;
  bool unsolvable_;
  TargetReadiness readiness_ = TargetReadiness::LOST;
  int last_track_id_ = -1;
  bool blind_ = false;
  bool has_start_timestamp_ = false;
  bool has_measurement_timestamp_ = false;
  bool has_full_observation_timestamp_ = false;
  std::chrono::steady_clock::time_point start_timestamp_{};
  std::chrono::steady_clock::time_point last_measurement_timestamp_{};
  std::chrono::steady_clock::time_point last_full_observation_timestamp_{};
  BuffPoseQuality last_pose_quality_ = BuffPoseQuality::FULL_8_POINT;
  int reset_count_ = 0;
  int innovation_reject_count_ = 0;
  bool has_plane_basis_ = false;
  Eigen::Vector3d plane_normal_{1.0, 0.0, 0.0};
  Eigen::Vector3d plane_normal_sum_{0.0, 0.0, 0.0};
  double plane_normal_weight_ = 0.0;
  Eigen::Vector3d phase_zero_axis_{0.0, 0.0, 1.0};
  Eigen::Vector3d phase_quarter_axis_{0.0, -1.0, 0.0};
  bool has_pending_phase_recovery_ = false;
  double pending_phase_recovery_ = 0.0;
  int pending_phase_recovery_hits_ = 0;
};

/// SmallTarget子类

class SmallTarget : public Target
{
public:
  /** @brief 使用默认配置构造小符目标 */
  SmallTarget();
  /** @brief 使用指定配置构造小符目标 @param config 能量机关配置 */
  explicit SmallTarget(BuffConfig config);
  /** @brief 从 YAML 配置构造小符目标 @param config_path 配置路径 */
  explicit SmallTarget(const std::string & config_path);

  using Target::get_target;

  /** @brief 使用可选观测更新小符目标 @param p 三维观测 @param timestamp 观测时间戳 */
  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  /** @brief 将小符相位向前预测 @param dt 预测时长，单位 s */
  void predict(double dt) override;

  /** @brief 克隆小符目标 @return 独立副本 */
  std::unique_ptr<Target> clone() const override { return std::make_unique<SmallTarget>(*this); }

  /** @brief 重置小符滤波和方向状态 */
  void reset() override;

private:
  /** @brief 使用首次观测初始化小符目标 @param nowtime 相对时间 @param p 三维观测 */
  void init(double nowtime, const PowerRune & p) override;

  /** @brief 使用观测更新小符目标 @param nowtime 相对时间 @param p 三维观测 */
  void update(double nowtime, const PowerRune & p) override;

  /** @brief 获取小符预测使用的旋转方向 @return 方向符号 */
  int small_prediction_roll_direction() const;

  /** @brief 判断小符方向是否稳定 @return 稳定时返回 true */
  bool has_stable_small_prediction_direction() const;

  const double SMALL_W = CV_PI / 3;
  // const double SMALL_W = 0;
  PhaseDirectionTracker phase_direction_;
};

/// BigTarget子类

class BigTarget : public Target
{
public:
  /** @brief 使用默认配置构造大符目标 */
  BigTarget();
  /** @brief 使用指定配置构造大符目标 @param config 能量机关配置 */
  explicit BigTarget(BuffConfig config);
  /** @brief 从 YAML 配置构造大符目标 @param config_path 配置路径 */
  explicit BigTarget(const std::string & config_path);

  using Target::get_target;

  /** @brief 使用可选观测更新大符目标 @param p 三维观测 @param timestamp 观测时间戳 */
  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  /** @brief 将大符相位向前预测 @param dt 预测时长，单位 s */
  void predict(double dt) override;

  /** @brief 克隆大符目标 @return 独立副本 */
  std::unique_ptr<Target> clone() const override { return std::make_unique<BigTarget>(*this); }

  /** @brief 重置大符滤波、拟合和方向状态 */
  void reset() override;

private:
  struct PhaseSample
  {
    double time = 0.0;
    double phase = 0.0;
    double weight = 1.0;
  };

  /** @brief 使用首次观测初始化大符目标 @param nowtime 相对时间 @param p 三维观测 */
  void init(double nowtime, const PowerRune & p) override;

  /** @brief 使用观测更新大符目标 @param nowtime 相对时间 @param p 三维观测 */
  void update(double nowtime, const PowerRune & p) override;

  /** @brief 清空速度样本 @param clear_fitter 是否同时清空 RANSAC 拟合器 */
  void clear_speed_samples(bool clear_fitter);

  /** @brief 根据滑动窗口估计角速度 @return 样本充分时返回角速度 */
  std::optional<double> estimate_window_speed() const;

  /** @brief 添加相位速度样本 @param nowtime 相对时间 @param observed_phase 观测相位 @param p 三维观测 */
  void add_speed_sample(double nowtime, double observed_phase, const PowerRune & p);

  tools::RansacSineFitter spd_fitter_;
  PhaseDirectionTracker phase_direction_;

  double fit_spd_ = 1.1775;
  double fit_blend_ = 0.0;
  double last_accepted_speed_time_ = 0.0;
  double last_fitter_sample_time_ = -1.0;
  double pause_speed_samples_until_ = 0.0;
  int speed_model_direction_ = 0;
  bool has_speed_center_ = false;
  Eigen::Vector3d last_speed_center_{0.0, 0.0, 0.0};
  std::deque<PhaseSample> phase_samples_;
  std::deque<double> accepted_speed_samples_;
};

}  // namespace auto_buff
#endif
