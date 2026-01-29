#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 26赛季前哨站参数
constexpr double TOWER_H_STEP = 0.1;        // 相邻装甲板高度差 0.1m
constexpr double TOWER_H_JUMP_THRES = 0.16; // 大跳变阈值

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

  // ============ [恢复 Q 矩阵参数] ============
  // 这里完全使用你原有的逻辑，不修改数值
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   
    v2 = 200;  // 保持原有的大 Yaw 轴噪声，允许快速旋转
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
// 1. 修改 update 函数：去掉过于敏感的重置逻辑，或者让它更宽容
void Target::update(const Armor & armor)
{
  // 1. 前哨站逻辑
  if (name == ArmorName::outpost) {
    updateTowerInfo(armor);
  }

  int id = 0;

  // 2. 确定装甲板 ID
  // 先跑一遍几何匹配，作为底板
  auto min_score = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
      xyza_i_list.push_back({xyza_list[i], i});
  }
  std::sort(
      xyza_i_list.begin(), xyza_i_list.end(),
      [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
        return a.first[3] < b.first[3];
      });

  for (int i = 0; i < armor_num_; i++) {
      const auto & xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
      auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));

      if (name == ArmorName::outpost) {
          // 几何匹配时，高度权重适当给，但不要太大，以免初始匹配错误
          double h_diff = std::abs(armor.xyz_in_world[2] - xyza[2]);
          angle_error += h_diff * 2.0; 
      } else {
          angle_error += std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
      }

      if (angle_error < min_score) {
          id = xyza_i_list[i].second;
          min_score = angle_error;
      }
  }

  // [关键修改] ID 融合逻辑
  if (name == ArmorName::outpost) {
      if (tower_initialized_) {
          // 如果逻辑已锁定，优先使用逻辑计算的 ID
          // 计算预测偏差
          Eigen::Vector3d predicted_xyz = h_armor_xyz(ekf_.x, current_tower_armor_id_);
          Eigen::Vector3d predicted_ypd = tools::xyz2ypd(predicted_xyz);
          double yaw_diff = std::abs(tools::limit_rad(armor.ypd_in_world[0] - predicted_ypd[0]));
          
          // 放宽容忍度：允许 60 度左右的偏差 (1.0 rad)
          // 前哨站转得快时，预测偏差可能会变大，太敏感会导致频繁重置
          if (yaw_diff < 1.0) {
              id = current_tower_armor_id_;
          } else {
              // 偏差极大，说明可能跟丢了，退回几何匹配，但保留尝试
              tools::logger()->warn("[Tower] Mismatch! LogicID: {}, GeomID: {}", current_tower_armor_id_, id);
              // 这里我们选择信任几何ID，并重置 current_tower 为几何 ID
              // 但不急着把 tower_initialized_ 置为 false，除非连续多次错误（这里简化处理）
              current_tower_armor_id_ = id; 
              // 暂时不置 false，看看能不能救回来。如果想严格点可以置 false
              // tower_initialized_ = false; 
          }
      } else {
          // 未初始化时，直接同步几何 ID，为 solveTowerLogic 提供基础
          current_tower_armor_id_ = id;
      }
  }

  // 3. 切换判定
  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
  } else {
    is_switch_ = false;
  }
  
  if (id != 0) jumped = true;
  last_id = id;
  update_count_++;

  update_ypda(armor, id);
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
      
      // 只有EKF稳定追踪一段时间后才启用逻辑判断
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
    
    // 转速太低不判断，防止误判方向
    if (std::abs(ekf_.x[7]) < 0.2) return;

    // 逻辑修正：
    // 不再检查 tower_initialized_。只要高度变化符合物理特征，就更新 ID。
    // 这样即使初始化断了，一旦出现 0.1m 的变化，也能把 ID 接上。

    if (is_ccw) {
        // CCW: 0 -> 2 -> 1 -> 0
        // 0 -> 2: +0.2 (Big Rise)
        // 2 -> 1: -0.1 (Small Drop)
        // 1 -> 0: -0.1 (Small Drop)

        if (diff > 0.14) { 
            // 大幅上升，必须是 0->2
            current_tower_armor_id_ = 2;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CCW Locked: Big Rise (0->2)");
        } 
        else if (diff < -0.06 && diff > -0.14) {
            // 小幅下降，可能是 2->1 或 1->0
            // 无论当前是几，逆时针的小下降就是 ID 减 1 (模3)
            // (id + 2) % 3 等价于 减1
            int next_id = (current_tower_armor_id_ + 2) % 3;
            tools::logger()->info("[Tower] CCW Step: Drop ({:.1f} -> {:.1f}) ID:{}->{}", last_tower_z_, now_tower_z_, current_tower_armor_id_, next_id);
            current_tower_armor_id_ = next_id;
        }
    } 
    else {
        // CW: 0 -> 1 -> 2 -> 0
        // 0 -> 1: +0.1 (Small Rise)
        // 1 -> 2: +0.1 (Small Rise)
        // 2 -> 0: -0.2 (Big Drop)

        if (diff < -0.14) {
            // 大幅下降，必须是 2->0
            current_tower_armor_id_ = 0;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CW Locked: Big Drop (2->0)");
        }
        else if (diff > 0.06 && diff < 0.14) {
            // 小幅上升，可能是 0->1 或 1->2
            // 无论当前是几，顺时针的小上升就是 ID 加 1 (模3)
            int next_id = (current_tower_armor_id_ + 1) % 3;
            tools::logger()->info("[Tower] CW Step: Rise ({:.1f} -> {:.1f}) ID:{}->{}", last_tower_z_, now_tower_z_, current_tower_armor_id_, next_id);
            current_tower_armor_id_ = next_id;
        }
    }
}
void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  
  Eigen::VectorXd R_dig;
  // ============ [恢复 R 矩阵参数] ============
  // 恢复为你原来代码中的逻辑，不再动态调整噪声
  // if (name == ArmorName::outpost) {
  //     // 保持你原有的设定，或者参考平衡步兵的设定
  //     // 假设原代码对前哨站有特殊设定，这里给一个稳定的值
  //     // 如果你之前代码里这里有特定值，请改回那个值。
  //     // 下面这个是比较通用的稳健值：
  //     R_dig = Eigen::VectorXd{{4e-3, 4e-3, 1e-2, 2e-2}}; 
  // } else {
      R_dig = Eigen::VectorXd{{4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
                           log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  // }

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
    // 普通装甲板保持不变
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