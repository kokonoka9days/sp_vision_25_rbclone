#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 26赛季前哨站参数
constexpr double TOWER_H_STEP = 0.1;        // 相邻装甲板高度差 0.1m
constexpr double TOWER_H_JUMP_THRES = 0.15; // 判定大幅度跳变的阈值 (0.15m)

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

  // 旋转中心的坐标初步估计
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2]; // 初始Z暂时认为是当前板子高度

  if (name == ArmorName::outpost) {
    armor_num_ = 3;
    // 对于前哨站，如果第一帧看到的是高板，这里初始化可能会有偏差，依靠后续solveTowerLogic修正
  }

  // x vx y vy z vz a w r l h
  // 注意：对于前哨站，x[4](z) 表示0号装甲板(最低)的高度，x[10](h) 实际上是固定的0.1
  // 这里初始化给 0.1，后续在 h_jacobian 中将其导数置0，使其保持不变
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
    v2 = 200;  // 前哨站方向随机，给予较大的Yaw轴噪声允许转向
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
  // 1. 前哨站逻辑：记录高度并解算状态
  if (name == ArmorName::outpost) {
    updateTowerInfo(armor);
  }

  int id = 0;

  // 2. 确定装甲板 ID
  if (name == ArmorName::outpost && tower_initialized_) {
    // 策略：如果前哨站状态已初始化，完全信任内部状态机计算出的 ID
    // 这样可以避免 EKF 在高度阶梯变化时计算出错误的 Residual
    id = current_tower_armor_id_;
  } 
  else {
    // 策略：普通装甲板，或未初始化的前哨站，使用几何距离（角度+高度）匹配
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      xyza_i_list.push_back({xyza_list[i], i});
    }

    // 按 Yaw 排序
    std::sort(
      xyza_i_list.begin(), xyza_i_list.end(),
      [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
        return a.first[3] < b.first[3];
      });

    // 寻找最匹配的板
    for (int i = 0; i < armor_num_; i++) {
      const auto & xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
      
      auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
      
      // 前哨站增加高度权重的匹配
      if (name == ArmorName::outpost) {
        // x[4] 是基准高度，xyza[2] 是计算出的该ID板的高度
        double h_diff = std::abs(armor.xyz_in_world[2] - xyza[2]);
        angle_error += h_diff * 5.0; 
      } else {
        angle_error += std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
      }

      if (angle_error < min_angle_error) {
        id = xyza_i_list[i].second;
        min_angle_error = angle_error;
      }
    }
    
    // 未初始化时，暂时同步 ID，方便调试（虽然可能不准）
    if (name == ArmorName::outpost && !tower_initialized_) {
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

  // 4. EKF 更新
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
  
  // 阈值判断：只有变化大于 5cm 才认为是可能的切换，否则视为震动噪声
  if (std::abs(diff) > 0.05) {
      last_tower_z_ = now_tower_z_;
      now_tower_z_ = current_z;
      
      // 只有当 EKF 速度估计稳定后（防止初始速度为0无法判断方向），再进行逻辑解算
      if (update_count_ > 10) { 
          solveTowerLogic();
      }
  } else {
      // 低通滤波平滑高度
      now_tower_z_ = 0.6 * now_tower_z_ + 0.4 * current_z;
  }
}

void Target::solveTowerLogic()
{
    // 需要有效的上一帧数据
    if (last_tower_z_ == 0) return;

    double diff = now_tower_z_ - last_tower_z_;
    bool is_ccw = getTowerVyawPositive(); // CCW > 0, CW < 0
    
    // 转速保护：如果角速度过小，方向不可信，跳过
    if (std::abs(ekf_.x[7]) < 0.2) return;

    // ==================== 26赛季前哨站逻辑核心 ====================
    // 结构：0(低)->1(中)->2(高) 逆时针分布。
    //
    // Case 1: 逆时针旋转 (CCW, w > 0)
    // 视觉观测顺序：0 -> 2 -> 1 -> 0
    // 高度变化：
    //   0 -> 2: +0.2 (Big Rise, 大幅上升)
    //   2 -> 1: -0.1
    //   1 -> 0: -0.1
    //
    // Case 2: 顺时针旋转 (CW, w < 0)
    // 视觉观测顺序：0 -> 1 -> 2 -> 0
    // 高度变化：
    //   0 -> 1: +0.1
    //   1 -> 2: +0.1
    //   2 -> 0: -0.2 (Big Drop, 大幅下降)

    if (is_ccw) {
        // 逆时针逻辑
        if (diff > TOWER_H_JUMP_THRES) { 
            // 发现大幅上升 (0->2)
            current_tower_armor_id_ = 2;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CCW Locked: Big Rise (0->2)");
        } 
        else if (tower_initialized_ && diff < -0.05) {
            // 已初始化，且高度下降 (-0.1)，符合 2->1 或 1->0
            // 公式：(id + 2) % 3。 例: 2->1, 1->0
            current_tower_armor_id_ = (current_tower_armor_id_ + 2) % 3;
        }
    } 
    else {
        // 顺时针逻辑
        if (diff < -TOWER_H_JUMP_THRES) {
            // 发现大幅下降 (2->0)
            current_tower_armor_id_ = 0;
            tower_initialized_ = true;
            tools::logger()->info("[Tower] CW Locked: Big Drop (2->0)");
        }
        else if (tower_initialized_ && diff > 0.05) {
            // 已初始化，且高度上升 (+0.1)，符合 0->1 或 1->2
            current_tower_armor_id_ = (current_tower_armor_id_ + 1) % 3;
        }
    }
}

void Target::update_ypda(const Armor & armor, int id)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  
  Eigen::VectorXd R_dig;
  if (name == ArmorName::outpost) {
      // 前哨站: 高度(Z)非常重要且固定，给予高信任；Yaw 允许一定噪声
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
  int limit = (name == ArmorName::outpost) ? 15 : 5; // 前哨站需要更长收敛时间
  if (update_count_ > limit && !diverged()) {
      is_converged_ = true;
  }
  return is_converged_;
}

// 观测模型：计算指定ID装甲板的XYZ
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto r = x[8]; 
  
  if (name == ArmorName::outpost) {
    // 26赛季前哨站：
    // x[4] 定义为 ID=0 (最低板) 的高度
    // ID=1 => +0.1m, ID=2 => +0.2m
    auto armor_x = x[0] - r * std::cos(angle);
    auto armor_y = x[2] - r * std::sin(angle);
    auto armor_z = x[4] + id * TOWER_H_STEP; 
    return {armor_x, armor_y, armor_z};
  } else {
    // 普通/平衡装甲板
    auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
    auto current_r = (use_l_h) ? x[8] + x[9] : x[8];
    auto armor_x = x[0] - current_r * std::cos(angle);
    auto armor_y = x[2] - current_r * std::sin(angle);
    auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];
    return {armor_x, armor_y, armor_z};
  }
}

// 雅可比矩阵
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  
  if (name == ArmorName::outpost) {
    auto r = x[8];
    auto dx_da = r * std::sin(angle);
    auto dy_da = -r * std::cos(angle);
    auto dx_dr = -std::cos(angle);
    auto dy_dr = -std::sin(angle);
    
    // 由于我们把高度差作为常量 TOWER_H_STEP，不对 x[10] 求导
    // 且 z = z_base + const，所以 dz/dz_base = 1, dz/dh = 0
    auto dz_dh = 0.0;

    // clang-format off
    Eigen::MatrixXd H_armor_xyza{
      {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, 0,     0}, // x, y, z, yaw, r, h...
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
    // 普通装甲板
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