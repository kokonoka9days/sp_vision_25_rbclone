#include "trajectory.hpp"

#include <cmath>
#include <iostream>

namespace tools
{
constexpr double g = 9.7833;

//同济原弹道模型
TrajectoryV1::TrajectoryV1(const double v0, const double d, const double h)
{
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    unsolvable = true;
    return;
  }
  // std::cout<<"delta :"<<delta<<std::endl;

  unsolvable = false;
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;
}

//V2空气阻力弹道模型
constexpr double kBig = 0.000429838;
constexpr double mBig = 0.043;
constexpr double kSmall = 0.000067165;
constexpr double mSmall = 0.0032;

TrajectoryV2::TrajectoryV2(const double v0, const double d, const double h){
  double k, m;
  bool isBigBullet = v0 > 18 ? false : true;
   fly_time = 0.;
  if (isBigBullet) {
      k = kBig;
      m = mBig;
  } else {
      k = kSmall;
      m = mSmall;
  }
  
  double kv2m = k * v0 * v0 / m;
  double discriminant = v0 * v0 - 2 * kv2m * d;
  
  if (discriminant < 0) {
    unsolvable = true; // 无正数解
    return;
  }
  
  double t1 = (v0 + std::sqrt(discriminant)) / kv2m;
  double t2 = (v0 - std::sqrt(discriminant)) / kv2m;
  
  // 选择正的时间解
  if (t1 > 0 && t2 > 0) {
      fly_time =  (t1 < t2) ? t1 : t2;
      unsolvable = false; 
  } else if (t1 > 0) {
      fly_time =  t1;
      unsolvable = false; 
  } else if (t2 > 0) {
      fly_time =  t2;
      unsolvable = false; 
  }else{
    unsolvable = true; // 无正数解
    return ;
  }

  // 计算目标角度（考虑重力影响的sinθ值）
  double target_sin_angle = (h + 0.5 * g * fly_time * fly_time) / (v0 * fly_time);
  
  // 检查sin值是否在有效范围内 [-1, 1]
  if (std::abs(target_sin_angle) > 1.0) {
      unsolvable = true;
      return;
  }

    // 3. 计算发射角度
  pitch = std::asin(target_sin_angle);
  
  // // 4. 计算重力补偿角度
  // // 重力补偿角度 = 实际发射角度 - 无重力时的发射角度
  // double no_gravity_angle = std::asin(h / d); // 注意：需要检查d不为0
  // if (d == 0) {
  //     no_gravity_angle = 0;
  // }
  
  // weight_compensation_angle = pitch - no_gravity_angle;
  
  unsolvable = false;
}

// V3：考虑阻力方向速度变化的数值积分模型
TrajectoryV3::TrajectoryV3(const double v0, const double d, const double h)
{
  bool isBigBullet = v0 > 18 ? false : true;
  double k = isBigBullet ? 0.000429838 : 0.000067165;
  double m = isBigBullet ? 0.043 : 0.0032;
  double coef = k / m; // 阻力加速度系数

  // 1. 初始化
  double dt = 0.001;               // 仿真步长：1ms（步长越小越准，算力消耗略增）
  double pitch_guess = std::atan2(h, d); // 初始猜测角度：直接瞄准目标的直线角度
  double y_aim = h;                // 虚拟瞄准高度，初始为实际目标高度
  
  unsolvable = true;
  fly_time = 0.0;
  pitch = 0.0;

  // 2. 迭代求解 (最多允许 20 次迭代，通常 3~5 次就能收敛)
  for (int i = 0; i < 20; ++i) {
      double x = 0.0;
      double y = 0.0;
      double vx = v0 * std::cos(pitch_guess);
      double vy = v0 * std::sin(pitch_guess);
      double t = 0.0;

      // 欧拉法数值积分仿真单次飞行
      while (x < d && t < 3.0) { // 限制最大飞行时间 3 秒防死循环
          double v = std::sqrt(vx * vx + vy * vy); // 实时合速度
          
          // 阻力加速度分解到 x 和 y 轴（这里的核心是耦合了速度矢量）
          double ax = -coef * v * vx;
          double ay = -g - coef * v * vy;
          
          vx += ax * dt;
          vy += ay * dt;
          x += vx * dt;
          y += vy * dt;
          t += dt;
      }

      // 计算到达目标水平距离 d 时的竖直误差
      double error = h - y;

      // 如果落点误差小于 1mm，认为已经命中，结束解算
      if (std::abs(error) < 0.001) {
          unsolvable = false;
          pitch = pitch_guess;
          fly_time = t;
          return;
      }

      // 3. 迭代补偿核心逻辑
      // 如果打低了 (error > 0)，就抬高虚拟瞄准高度；反之亦然
      y_aim += error; 
      pitch_guess = std::atan2(y_aim, d);
  }
  
  // 如果 20 次迭代都没收敛，维持 unsolvable = true
}

}  // namespace tools