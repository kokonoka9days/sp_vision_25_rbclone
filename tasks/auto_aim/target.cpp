#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 26赛季前哨站参数
constexpr double TOWER_H_STEP = 0.1;        // 相邻装甲板高度差 0.1m
constexpr double TOWER_H_JUMP_THRES = 0.15; // 大跳变阈值

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
  current_tower_armor_id_(0),
  tower_initialized_(false),
  last_tower_z_(0),
  now_tower_z_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2]; 

  if (name == ArmorName::outpost) {
    armor_num_ = 3;
  }

  // x[4] (z) 为基准高度, x[10] (h) 为高度差常量 0.1
  double initial_h = (name == ArmorName::outpost) ? TOWER_H_STEP : 0.1;
  
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, initial_h}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
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
  // 状态转移矩阵 F
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

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   
    v2 = 200;  
  } else {
    v1 = 100;
    v2 = 400;
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
  Q.block<2, 2>(0, 0) = Eigen::Matrix2d{{a, b}, {b, c}} * v1; // X
  Q.block<2, 2>(2, 2) = Eigen::Matrix2d{{a, b}, {b, c}} * v1; // Y
  Q.block<2, 2>(4, 4) = Eigen::Matrix2d{{a, b}, {b, c}} * v1; // Z
  Q.block<2, 2>(6, 6) = Eigen::Matrix2d{{a, b}, {b, c}} * v2; // Yaw

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  // 1. 前哨站逻辑：记录高度并尝试解算状态
  if (name == ArmorName::outpost) {
    updateTowerInfo(armor);
  }

  // 2. 计算几何最佳 ID (id_geom)
  // 这是为了保证 EKF 的连续性，必须找一个角度最近的
  int id_geom = 0;
  auto min_score = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  
  // 寻找几何上最接近的ID
  for (int i = 0; i < armor_num_; i++) {
      auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza_list[i][3]));
      if (name == ArmorName::outpost) {
          // 几何匹配稍微带一点高度权重，防止纯重合
          double h_diff = std::abs(armor.xyz_in_world[2] - xyza_list[i][2]);
          angle_error += h_diff * 2.0; 
      } else {
          Eigen::Vector3d ypd = tools::xyz2ypd(xyza_list[i].head(3));
          angle_error += std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
      }

      if (angle_error < min_score) {
          id_geom = i;
          min_score = angle_error;
      }
  }

  // 3. 决定用于 EKF 更新的 ID (final_ekf_id)
  int final_ekf_id = id_geom;

  if (name == ArmorName::outpost) {
      if (!tower_initialized_) {
          // 未初始化时，逻辑ID跟随几何ID，先跑起来
          current_tower_armor_id_ = id_geom;
      } 
      else {
          // 已初始化，我们有一个 "逻辑上的 ID" (current_tower_armor_id_)
          // 我们需要检查这个逻辑 ID 是否符合几何约束
          
          Eigen::Vector3d predicted_xyz = h_armor_xyz(ekf_.x, current_tower_armor_id_);
          Eigen::Vector3d predicted_ypd = tools::xyz2ypd(predicted_xyz);
          double yaw_diff = std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));

          // [关键修复]
          // 如果逻辑 ID 的预测角度和观测角度偏差 < 0.6 rad (约35度)，
          // 说明逻辑 ID 是合理的，我们优先使用它来修正 EKF。
          // 否则，如果偏差太大，强制使用几何 ID 更新 EKF，防止模型炸裂，
          // 但保留 current_tower_armor_id_ 的值（相信逻辑判断，等待旋转到位）。
          if (yaw_diff < 0.6) {
              final_ekf_id = current_tower_armor_id_;
          } else {
              // 偏差过大，退回几何ID进行更新
              final_ekf_id = id_geom;
              // 此时不重置 tower_initialized_，因为可能是短暂的跟丢或延迟
          }
      }
  }

  // 4. 切换判定
  if (final_ekf_id != last_id) {
    is_switch_ = true;
    switch_count_++;
  } else {
    is_switch_ = false;
  }
  
  if (final_ekf_id != 0) jumped = true;
  last_id = final_ekf_id;
  update_count_++;

  // 5. EKF 更新
  // 无论逻辑判定如何，这里喂进去的一定是符合几何约束的 ID
  update_ypda(armor, final_ekf_id);
}

void Target::updateTowerInfo(const Armor & armor)
{
  double current_z = armor.xyz_in_world[2];

  if (last_tower_z_ == 0 && now_tower_z_ == 0) {
      now_tower_z_ = current_z;
      return; 
  }

  double diff = current_z - now_tower_z_;
  
  if (std::abs(diff) > 0.05) {
      last_tower_z_ = now_tower_z_;
      now_tower_z_ = current_z;
      
      if (update_count_ > 20) { 
          solveTowerLogic();
      }
  } else {
      now_tower_z_ = 0.6 * now_tower_z_ + 0.4 * current_z;
  }
}

void Target::solveTowerLogic()
{
    if (last_tower_z_ == 0) return;

    double diff = now_tower_z_ - last_tower_z_;
    bool is_ccw = getTowerVyawPositive(); 
    
    // [关键修复] 提高转速阈值
    // 防止低速震荡时方向误判导致逻辑反转
    if (std::abs(ekf_.x[7]) < 0.4) return;

    if (is_ccw) {
        // CCW: 0 -> 2 -> 1 -> 0
        if (diff > 0.14) { 
            // 0 -> 2 (Big Rise)
            current_tower_armor_id_ = 2;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CCW Locked: Rise (0->2)");
        } 
        else if (diff < -0.06 && diff > -0.14) {
            // 2 -> 1 or 1 -> 0 (Small Drop)
            // (id + 2) % 3 相当于 id - 1
            current_tower_armor_id_ = (current_tower_armor_id_ + 2) % 3;
            tools::logger()->info("[Tower] CCW Step: Drop -> ID {}", current_tower_armor_id_);
        }
    } 
    else {
        // CW: 0 -> 1 -> 2 -> 0
        if (diff < -0.14) {
            // 2 -> 0 (Big Drop)
            current_tower_armor_id_ = 0;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CW Locked: Drop (2->0)");
        }
        else if (diff > 0.06 && diff < 0.14) {
            // 0 -> 1 or 1 -> 2 (Small Rise)
            // (id + 1) % 3
            current_tower_armor_id_ = (current_tower_armor_id_ + 1) % 3;
            tools::logger()->info("[Tower] CW Step: Rise -> ID {}", current_tower_armor_id_);
        }
    }
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  
  Eigen::VectorXd R_dig;
  // 保持原有观测噪声参数
  if (name == ArmorName::outpost) {
      R_dig = Eigen::VectorXd{{4e-3, 4e-3, 1e-2, 2e-2}}; 
  } else {
      R_dig = Eigen::VectorXd{{4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
                           log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  }

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

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
  auto r_ok = ekf_.x[8] > 0.1 && ekf_.x[8] < 0.6;
  if (!r_ok) return true;
  return false;
}

bool Target::convergened()
{
  int limit = (name == ArmorName::outpost) ? 15 : 5; 
  if (update_count_ > limit && !diverged()) {
      is_converged_ = true;
  }
  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto r = x[8]; 
  
  if (name == ArmorName::outpost) {
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    auto armor_z = x[4] + id * TOWER_H_STEP; 
    return {armor_x, armor_y, armor_z};
  } else {
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto current_r = (use_l_h) ? x[8] + x[9] : x[8];
    auto armor_x = x[0] - current_r * std::cos(angle);
    auto armor_y = x[2] - current_r * std::sin(angle);
    auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];
    return {armor_x, armor_y, armor_z};
  }
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  
  if (name == ArmorName::outpost) {
    auto r = x[8];
    auto dx_da = r * std::sin(angle);
    auto dy_da = -r * std::cos(angle);
    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    auto dz_dh = 0.0;

    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, 0,     0}, 
      {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, 0,     0},
      {0, 0, 0, 0, 1, 0,     0, 0,     0, 0, dz_dh},
      {0, 0, 0, 0, 0, 0,     1, 0,     0, 0,     0}
    };
    // clang-format on

    Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
    Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
    Eigen::MatrixXd H_armor_ypda = Eigen::MatrixXd::Identity(4, 4);
    H_armor_ypda.block<3, 3>(0, 0) = H_armor_ypd;

    return H_armor_ypda * H_armor_xyza;
  } else {
    // 普通装甲板部分
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
    Eigen::MatrixXd H_armor_ypda{
      {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
      {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
      {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
      {                0,                 0,                 0, 1}
    };
    return H_armor_ypda * H_armor_xyza;
  }
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim