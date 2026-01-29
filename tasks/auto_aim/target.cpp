#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0),
  is_outpost_(armor.name == ArmorName::outpost),
  fixed_radius_(0.0),
  fixed_omega_(0.0),
  omega_sign_(1.0)  // 默认正向旋转
{
  // 前哨站特殊处理
  if (is_outpost_) {
    armor_num_ = 3;  // 固定为3块装甲板
    
    // 初始化高度偏移：从低到高，逆时针依次增高
    height_offsets_ = {0.0, 0.10, 0.20};  // 单位：米
    
    // 固定半径（根据实际测量值设置，假设为0.3米）
    fixed_radius_ = 0.3;
    
    // 固定角速度（根据实际测量值设置，假设为2.0 rad/s）
    fixed_omega_ = 2.0;
    
    // 初始化状态向量，前哨站不需要估计半径和角速度
    const Eigen::VectorXd & xyz = armor.xyz_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;
    
    // 旋转中心坐标
    auto center_x = xyz[0] + fixed_radius_ * std::cos(ypr[0]);
    auto center_y = xyz[1] + fixed_radius_ * std::sin(ypr[0]);
    auto center_z = xyz[2] - height_offsets_[0];  // 减去第一个装甲板的高度偏移
    
    // 状态向量：[x, vx, y, vy, z, vz, a, 0, 0, 0, 0]
    // 前哨站的半径和角速度固定，不需要作为状态量
    Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, 0, 0, 0}};
    
    // 调整协方差矩阵，将固定参数的方差设小
    P0_dig[7] = 0.001;  // 角速度方差小
    P0_dig[8] = 0.001;  // 半径方差小
    P0_dig[9] = 0.001;  // l方差小
    P0_dig[10] = 0.001; // h方差小
    
    Eigen::MatrixXd P0 = P0_dig.asDiagonal();
    
    auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a + b;
      c[6] = tools::limit_rad(c[6]);
      return c;
    };
    
    ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  }
  else {
    // 原有逻辑保持不变
    auto r = radius;
    priority = armor.priority;
    const Eigen::VectorXd & xyz = armor.xyz_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;
    
    auto center_x = xyz[0] + r * std::cos(ypr[0]);
    auto center_y = xyz[1] + r * std::sin(ypr[0]);
    auto center_z = xyz[2];
    
    Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
    Eigen::MatrixXd P0 = P0_dig.asDiagonal();
    
    auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a + b;
      c[6] = tools::limit_rad(c[6]);
      return c;
    };
    
    ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  }
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();
  
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };
  
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 前哨站特殊处理
  if (is_outpost_) {
    // 使用固定角速度
    double omega = fixed_omega_ * omega_sign_;
    
    // 状态转移矩阵（前哨站简化版）
    // clang-format off
    Eigen::MatrixXd F{
      {1, dt,  0,  0,  0,  0,        0,  0, 0, 0, 0},
      {0,  1,  0,  0,  0,  0,        0,  0, 0, 0, 0},
      {0,  0,  1, dt,  0,  0,        0,  0, 0, 0, 0},
      {0,  0,  0,  1,  0,  0,        0,  0, 0, 0, 0},
      {0,  0,  0,  0,  1, dt,        0,  0, 0, 0, 0},
      {0,  0,  0,  0,  0,  1,        0,  0, 0, 0, 0},
      {0,  0,  0,  0,  0,  0,        1, dt, 0, 0, 0},
      {0,  0,  0,  0,  0,  0,        0,  1, 0, 0, 0},
      {0,  0,  0,  0,  0,  0,        0,  0, 1, 0, 0},
      {0,  0,  0,  0,  0,  0,        0,  0, 0, 1, 0},
      {0,  0,  0,  0,  0,  0,        0,  0, 0, 0, 1}
    };
    // clang-format on
    
    // 固定角速度，将角速度状态设为固定值
    ekf_.x[7] = omega;
    
    // 前哨站过程噪声较小
    double v1 = 5.0;   // 前哨站加速度方差更小
    double v2 = 0.01;  // 前哨站角加速度方差非常小
    
    auto a = dt * dt * dt * dt / 4;
    auto b = dt * dt * dt / 2;
    auto c = dt * dt;
    
    // clang-format off
    Eigen::MatrixXd Q{
      {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
    };
    // clang-format on
    
    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = tools::limit_rad(x_prior[6]);
      // 保持固定角速度
      x_prior[7] = omega;
      return x_prior;
    };
    
    ekf_.predict(F, Q, f);
  }
  else {
    // 状态转移矩阵
    // clang-format off
    Eigen::MatrixXd F{
      {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
    };
    // clang-format on

    // Piecewise White Noise Model
    double v1, v2;
    if (name == ArmorName::outpost) {
      v1 = 10;   // 前哨站加速度方差
      v2 = 0.1;  // 前哨站角加速度方差
    } else {
      v1 = 100;  // 加速度方差
      v2 = 400;  // 角加速度方差
    }
    auto a = dt * dt * dt * dt / 4;
    auto b = dt * dt * dt / 2;
    auto c = dt * dt;
    // 预测过程噪声偏差的方差
    // clang-format off
    Eigen::MatrixXd Q{
      {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
      {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
    };
    // clang-format on

    // 防止夹角求和出现异常值
    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = tools::limit_rad(x_prior[6]);
      return x_prior;
    };

    // 前哨站转速特判（注意：这里前哨站已经单独处理，但为了兼容原代码，保留此判断）
    if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
      this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

    ekf_.predict(F, Q, f);
  }
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 前哨站匹配策略调整
  int check_count = std::min(3, armor_num_);
  for (int i = 0; i < check_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    
    // 前哨站增加高度匹配误差
    double height_error = 0.0;
    if (is_outpost_) {
      height_error = std::abs(armor.xyz_in_world[2] - xyza[2]);
    }
    
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0])) +
                       0.5 * height_error;  // 高度误差权重
    
    if (angle_error < min_angle_error) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }
  
  // 检测旋转方向（前哨站）
  if (is_outpost_ && update_count_ == 1) {
    // 根据第二帧的角度变化判断旋转方向
    double angle_diff = tools::limit_rad(armor.ypr_in_world[0] - ekf_.x[6]);
    // double dt = tools::delta_time(t_, armor.detected_time);
    double measured_omega = angle_diff;// / dt;
    
    if (measured_omega < 0) {
      omega_sign_ = -1.0;
    } else {
      omega_sign_ = 1.0;
    }
  }
  
  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  // 前哨站特殊处理
  if (is_outpost_) {
    // 前哨站使用固定半径，只需要检查位置合理性
    auto center_x = ekf_.x[0];
    auto center_y = ekf_.x[2];
    auto center_z = ekf_.x[4];
    
    // 检查中心坐标是否在合理范围内
    bool position_ok = std::abs(center_x) < 10.0 && 
                       std::abs(center_y) < 10.0 && 
                       std::abs(center_z) < 5.0;
    
    // 检查角度是否正常
    bool angle_ok = std::abs(ekf_.x[6]) < CV_PI * 2;
    
    return !(position_ok && angle_ok);
  }
  else {
    // 原有逻辑
    auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
    auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

    if (r_ok && l_ok) return false;

    return true;
  }
}

bool Target::convergened()
{
  if (is_outpost_) {
    // 前哨站收敛更快，因为参数固定
    if (update_count_ > 3 && !this->diverged()) {
      is_converged_ = true;
    }
  }
  else if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断（原代码，但我们已经在前面的if分支处理了前哨站）
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  // 前哨站特殊处理
  if (is_outpost_) {
    double angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    double r = fixed_radius_;
    
    // 使用固定半径计算坐标
    double armor_x = x[0] - r * std::cos(angle);
    double armor_y = x[2] - r * std::sin(angle);
    
    // 使用预设的高度偏移
    double armor_z = x[4] + height_offsets_[id];
    
    return {armor_x, armor_y, armor_z};
  }
  else {
    // 原有逻辑
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

    auto r = (use_l_h) ? x[8] + x[9] : x[8];
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

    return {armor_x, armor_y, armor_z};
  }
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  if (is_outpost_) {
    double angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    double r = fixed_radius_;
    
    double dx_da = r * std::sin(angle);
    double dy_da = -r * std::cos(angle);
    
    // 前哨站的半径固定，不进行估计
    double dx_dr = 0.0;
    double dy_dr = 0.0;
    double dx_dl = 0.0;
    double dy_dl = 0.0;
    
    // 高度是固定的偏移，不随状态变化
    double dz_dh = 0.0;
    
    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
      {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
      {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
      {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
    };
    // clang-format on
    
    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    
    // clang-format off
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {                0,                 0,                 0, 1}
    };
    // clang-format on
    
    return H_armor_ypda * H_armor_xyza;
  }
  else {
    // 原有逻辑
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

    auto r = (use_l_h) ? x[8] + x[9] : x[8];
    auto dx_da = r * std::sin(angle);
    auto dy_da = -r * std::cos(angle);

    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
    auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

    auto dz_dh = (use_l_h) ? 1.0 : 0.0;

    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
      {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
      {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
      {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
    };
    // clang-format on

    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    // clang-format off
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {                0,                 0,                 0, 1}
    };
    // clang-format on

    return H_armor_ypda * H_armor_xyza;
  }
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim