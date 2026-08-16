#ifndef TOOLS__PID_HPP
#define TOOLS__PID_HPP

namespace tools
{
class PID
{
public:
  /**
   * @brief 构造 PID 控制器
   * @param dt 控制周期，单位 s
   * @param kp 比例系数
   * @param ki 积分系数
   * @param kd 微分系数
   * @param max_out 输出绝对值上限
   * @param max_iout 积分项绝对值上限
   * @param angular 是否将误差按周期角处理
   */
  PID(float dt, float kp, float ki, float kd, float max_out, float max_iout, bool angular = false);

  float pout = 0.0f;  // P项输出, 用于调试
  float iout = 0.0f;  // I项输出, 用于调试
  float dout = 0.0f;  // D项输出, 用于调试

  /** @brief 计算 PID 输出 @param set 目标值 @param fdb 反馈值 @return 限幅后的控制量 */
  float calc(float set, float fdb);

private:
  const float dt_;
  const float kp_, ki_, kd_;
  const float max_out_, max_iout_;
  const bool angular_;

  float last_fdb_ = 0.0f;  // 上次反馈值
};

}  // namespace tools

#endif  // TOOLS__PID_HPP
