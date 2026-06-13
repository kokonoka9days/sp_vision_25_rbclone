#include "target.hpp"

#include <numeric>
#include <cmath>
#include <algorithm>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 物理常量定义
constexpr double TOWER_ARMOR_DH = 0.10;  // 前哨站两个装甲板之间的标准高低差(m)
constexpr double TOWER_ARMOR_DTB = 0.16;  // 前哨装甲大跳变阈值(m)
constexpr double TOWER_ARMOR_XTB = 0.05;  // 前哨装甲小跳变阈值(m)

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
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 根据当前装甲板位置和半径，反推旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  if(name == ArmorName::outpost){
    tower_armor_hs[0].first = true;       // 标记 0 号位已成功初始化
    tower_armor_hs[0].second = center_z;  // 记录真实高度
  }

  cam_is_switch_time_point = std::chrono::steady_clock::time_point{};

  // ==========================================
  // EKF 11维状态向量定义:
  // [0]x, [1]vx, [2]y, [3]vy, [4]z, [5]vz, 
  // [6]yaw(偏航角), [7]vyaw(自转角速度), 
  // [8]r(基础半径), [9]r_(半径补偿量), [10]z_(高度补偿量)
  // ==========================================
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);

  // 如果是前哨站，将 z_ (x[10]) 的初始值设为物理理论值
  double initial_dz = (name == ArmorName::outpost) ? TOWER_ARMOR_DH : 0.0;

  x0 << center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, initial_dz;
  
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 10.0;
  P0.block(0, 0, 11, 11) = P0_dig.asDiagonal();

  // 自定义状态加法，确保角度(Yaw)在 -PI 到 PI 之间
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

// 供手动初始化使用的构造函数
Target::Target(double x, double vyaw, double radius, double h) 
: armor_num_(4)
{
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(11);
  x0 << x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h;

  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 10.0;
  P0.block(0, 0, 11, 11) = P0_dig.asDiagonal();

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
  // ================= 1. 原有的整车 EKF (CV模型) 预测 =================
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }

  auto a_ = dt * dt * dt * dt / 4;
  auto b_ = dt * dt * dt / 2;
  auto c_ = dt * dt;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
  Q(0,0) = a_ * v1; Q(0,1) = b_ * v1; Q(1,0) = b_ * v1; Q(1,1) = c_ * v1; // X
  Q(2,2) = a_ * v1; Q(2,3) = b_ * v1; Q(3,2) = b_ * v1; Q(3,3) = c_ * v1; // Y
  Q(4,4) = a_ * v1; Q(4,5) = b_ * v1; Q(5,4) = b_ * v1; Q(5,5) = c_ * v1; // Z
  Q(6,6) = a_ * v2; Q(6,7) = b_ * v2; Q(7,6) = b_ * v2; Q(7,7) = c_ * v2; // Yaw

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站收敛后限制最大转速防飞
  if (this->convergened() && this->name == ArmorName::outpost) {
    if (std::abs(this->ekf_.x[7]) > 2) this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

 ekf_.predict(F, Q, f);

  // ================= 2. 中心级 CA EKF 预测 =================
  if (ca_ekf_init_) {
    // F_CA: 9x9 完整 CA 模型 (位置←速度←加速度)
    Eigen::MatrixXd F_CA = Eigen::MatrixXd::Identity(9, 9);
    for (int i = 0; i < 3; i++) {
      F_CA(i*3, i*3+1) = dt;
      F_CA(i*3+1, i*3+2) = dt;
      F_CA(i*3, i*3+2) = 0.5 * dt * dt;
    }

    // Q_CA: Singer 型块对角，q=1.0
    double q = 1.0;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;
    Eigen::MatrixXd Q_CA = Eigen::MatrixXd::Zero(9, 9);
    for (int i = 0; i < 3; i++) {
      int j = i * 3;
      Q_CA(j,   j)   = q * dt5 / 20.0;  Q_CA(j,   j+1) = q * dt4 / 8.0;   Q_CA(j,   j+2) = q * dt3 / 6.0;
      Q_CA(j+1, j)   = q * dt4 / 8.0;   Q_CA(j+1, j+1) = q * dt3 / 3.0;   Q_CA(j+1, j+2) = q * dt2 / 2.0;
      Q_CA(j+2, j)   = q * dt3 / 6.0;   Q_CA(j+2, j+1) = q * dt2 / 2.0;   Q_CA(j+2, j+2) = q * dt;
    }

    ca_ekf_.predict(F_CA, Q_CA);
  }


  // ================= 3. 计算Singer思想的机动频率权重 =================
  double vyaw = std::abs(ekf_.x[7]);
  if (this->name == ArmorName::outpost) {
      // 前哨站强制全信任整车CV
      w_cv_ = 1.0; 
  } else {
      constexpr double OMEGA_THRESH = 1.5; // 机动频率融合阈值 (rad/s)
      w_cv_ = 1.0 - std::exp(-vyaw / OMEGA_THRESH);
  }
}

void Target::update(const Armor & armor)
{
  int id = 0;

  if (this->name == ArmorName::outpost) {
    // 【策略 A：前哨站专用】
    // 纯几何匹配(距离+复合角度)，绕开因高度阶梯跳变导致 EKF 协方差波动的干扰
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

    this->ekf_x()(10) = TOWER_ARMOR_DH;

    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      xyza_i_list.push_back({xyza_list[i], i});
    }

    // 按距离(ypd[2])由近及远排序
    std::sort(
      xyza_i_list.begin(), xyza_i_list.end(),
      [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
        Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
        Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
        return ypd1[2] < ypd2[2];
      });

    // 只取最近的3个装甲板验证角度匹配度
    for (int i = 0; i < 3; i++) {
      const auto & xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
      
      auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                         std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

      if (std::abs(angle_error) < std::abs(min_angle_error)) {
        id = xyza_i_list[i].second;
        min_angle_error = angle_error;
      }
    }

  } else {
    // 【策略 B：其他兵种通用】
    // 马氏距离匹配 + 迟滞防抖，有效应对平移带来的透视形变
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

    auto r2_azimuth = 4e-3;
    auto r2_pitch = 4e-3;
    auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
    auto r2_d = log(std::abs(delta_angle) + 1) + 1;
    
    // 处理镜头长短焦切换时的噪声激增
    if (last_cam_is_short != cam_is_short) {
      cam_is_switch_time_point = std::chrono::steady_clock::now();
      last_cam_is_short = cam_is_short;
    }
    auto now = std::chrono::steady_clock::now();
    double cam_is_switch_lter_dt = tools::delta_time(now, cam_is_switch_time_point);
    if (cam_is_switch_lter_dt < 0.7 && update_count_ > 50) {
      r2_azimuth = 4e+4;
      r2_angle *= 300;
      r2_d *= 300;
    }
    
    Eigen::VectorXd R_dig{{r2_azimuth, r2_pitch, r2_d, r2_angle}};
    Eigen::MatrixXd R = R_dig.asDiagonal();

    const Eigen::VectorXd & ypd = armor.ypd_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;
    Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

    int best_id = 0;
    double min_mahalanobis_dist = 1e10;
    std::vector<double> md_list(armor_num_, 1e10);

    for (int i = 0; i < armor_num_; i++) {
      Eigen::VectorXd xyz_pred = h_armor_xyz(ekf_.x, i);
      Eigen::VectorXd ypd_pred = tools::xyz2ypd(xyz_pred);
      auto angle_pred = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
      Eigen::VectorXd z_pred{{ypd_pred[0], ypd_pred[1], ypd_pred[2], angle_pred}};

      Eigen::VectorXd y = z - z_pred;
      y[0] = tools::limit_rad(y[0]);
      y[1] = tools::limit_rad(y[1]);
      y[3] = tools::limit_rad(y[3]);

      Eigen::MatrixXd H = h_jacobian(ekf_.x, i);
      Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;

      double mahalanobis_dist = y.transpose() * S.inverse() * y;
      md_list[i] = mahalanobis_dist;

      if (mahalanobis_dist < min_mahalanobis_dist) {
        min_mahalanobis_dist = mahalanobis_dist;
        best_id = i;
      }
    }

    // 迟滞防抖动（Hysteresis）机制：倾向于保持上一次匹配的ID
    id = best_id;
    double CHI_SQ_THRESHOLD = 9.488; 
    double HYSTERESIS_MARGIN = 5.0; 

    if (md_list[last_id] < CHI_SQ_THRESHOLD) {
      if (min_mahalanobis_dist > md_list[last_id] - HYSTERESIS_MARGIN) {
        id = last_id;
      }
    }
  }

  if (id != 0) jumped = true;

  // 检测换板事件
  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
    
    // 换板时，将上一块装甲板的历史累加数据计算为平均高度锚点
    if (name == ArmorName::outpost) {
      if (tower_armor_hs_datas_ptr[last_id] > 0) {
        tower_armor_hs[last_id].first = true; // 标记该装甲板已有有效的历史数据
        tower_armor_hs[last_id].second = tower_armor_hs_datas[last_id] / tower_armor_hs_datas_ptr[last_id];
      }
    }
  } else {
    is_switch_ = false;
  }

  // 累加当前块装甲板的高度特征
  if(name == ArmorName::outpost){
    double a = 0.1; // 互补滤波系数
    tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
    
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr[id]++;     

    // 历史高度数据保护机制，防止长时间追踪导致累加溢出
    if(tower_armor_hs_datas[id] > 10000){
      tower_armor_hs_datas[id] = (tower_armor_hs_datas[id] / tower_armor_hs_datas_ptr[id]) * 600;
      tower_armor_hs_datas_ptr[id] = 600;
    }
  }

  last_id = id;
  update_count_++;    
  xyz_in_world = armor.xyz_in_world;

  // ================= 1. EKF 状态更新 =================
  update_ypda(armor, id);

  // ================= 2. 中心级 CA EKF 更新 =================
  {
    // 从观测板位反算中心 (plate observation -> center)
    double c = std::cos(armor.ypr_in_world[0]);
    double s = std::sin(armor.ypr_in_world[0]);
    Eigen::Vector3d z_ca(
      armor.xyz_in_world[0] + ekf_.x[8] * c,
      armor.xyz_in_world[1] + ekf_.x[8] * s,
      armor.xyz_in_world[2]
    );

    if (!ca_ekf_init_) {
      // 惰性初始化: 位置 = 观测中心, 速度 = CV EKF 当前速度, 加速度 = 0
      Eigen::VectorXd x0_ca = Eigen::VectorXd::Zero(9);
      x0_ca(0) = z_ca(0);  x0_ca(1) = ekf_.x[1];  x0_ca(2) = 0;
      x0_ca(3) = z_ca(1);  x0_ca(4) = ekf_.x[3];  x0_ca(5) = 0;
      x0_ca(6) = z_ca(2);  x0_ca(7) = ekf_.x[5];  x0_ca(8) = 0;

      Eigen::MatrixXd P0_ca = Eigen::MatrixXd::Identity(9, 9) * 10.0;
      auto x_add_ca = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; };
      ca_ekf_ = tools::ExtendedKalmanFilter(x0_ca, P0_ca, x_add_ca);
      ca_ekf_init_ = true;
    } else {
      // 6维观测: 位置(从板位反算的中心) + CV速度作为速度观测
      Eigen::VectorXd z_ca_ext(6);
      z_ca_ext << z_ca, ekf_.x[1], ekf_.x[3], ekf_.x[5];

      auto h_ca = [](const Eigen::VectorXd & x) -> Eigen::VectorXd {
        Eigen::VectorXd z_pred(6);
        z_pred << x[0], x[3], x[6], x[1], x[4], x[7];
        return z_pred;
      };
      auto z_sub_ca = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; };

      Eigen::MatrixXd H_ca = Eigen::MatrixXd::Zero(6, 9);
      H_ca(0, 0) = 1;  H_ca(1, 3) = 1;  H_ca(2, 6) = 1;   // 位置观测
      H_ca(3, 1) = 1;  H_ca(4, 4) = 1;  H_ca(5, 7) = 1;   // 速度观测

      // 位置噪声: 0.01·I (σ≈0.1m), 速度噪声: 0.2·I (σ≈0.45m/s, 偏保守)
      Eigen::MatrixXd R_ca = Eigen::MatrixXd::Zero(6, 6);
      R_ca(0,0) = R_ca(1,1) = R_ca(2,2) = 0.01;
      R_ca(3,3) = R_ca(4,4) = R_ca(5,5) = 0.2;

      ca_ekf_.update(z_ca_ext, H_ca, R_ca, h_ca, z_sub_ca);
    }
  }
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  // 用距离判断观测数据，距离越远观测越不可信
  double distance = std::abs(armor.ypd_in_world[2]);
  double dist_penalty = distance * distance * 0.05;

  auto r2_azimuth = 4e-3 + distance * 1e-3;
  auto r2_pitch   = 4e-3 + distance * 1e-3;
  auto r2_angle   = log(distance + 1) / 200 + 9e-2 + dist_penalty * 0.1;
  auto r2_d       = log(std::abs(delta_angle) + 1) + 1 + dist_penalty;
  
  if(last_cam_is_short != cam_is_short){
    cam_is_switch_time_point = std::chrono::steady_clock::now();
    last_cam_is_short = cam_is_short;
  }
  auto now = std::chrono::steady_clock::now();
  double cam_is_switch_lter_dt = tools::delta_time(now, cam_is_switch_time_point);
  if(cam_is_switch_lter_dt < 0.7 && update_count_ > 50){
    r2_azimuth = 4e+4;
    r2_angle *= 300;
    r2_d *= 300;
  }
  
  Eigen::VectorXd R_dig{{r2_azimuth, r2_pitch, r2_d, r2_angle}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 预测观测函数 h(x)
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 自定义减法（处理角度越界）
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

  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  ekf_.update(z, H, R, h, z_subtract);

  if (update_count_ < 15) {
    double damping = update_count_ / 15.0; // 从 1/15 线性平滑增加到 1.0
    ekf_.x[1] *= damping; // vx 阻尼
    ekf_.x[3] *= damping; // vy 阻尼
    ekf_.x[5] *= damping; // vz 阻尼
    ekf_.x[7] *= damping; // vyaw (角速度) 阻尼
  }
}

// 获取 EKF 状态向量
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

// 获取滤波器常引用
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

// 返回所有装甲板的预测四维状态 (X, Y, Z, Angle) 列表
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  double w_ca = 1.0 - w_cv_;

  Eigen::Vector3d center_CV(ekf_.x[0], ekf_.x[2], ekf_.x[4]);

  Eigen::Vector3d center_fused;
  if (ca_ekf_init_ && w_ca > 0.05) {
    Eigen::Vector3d center_CA(ca_ekf_.x[0], ca_ekf_.x[3], ca_ekf_.x[6]);
    center_fused = w_cv_ * center_CV + w_ca * center_CA;
  } else {
    center_fused = center_CV;
  }

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    auto use_l_h = (armor_num_ == 4) && (i == 1 || i == 3);

    double r = (use_l_h) ? ekf_.x[8] + ekf_.x[9] : ekf_.x[8];
    double plate_x = center_fused.x() - r * std::cos(angle);
    double plate_y = center_fused.y() - r * std::sin(angle);

    double plate_z;
    if (name == ArmorName::outpost) {
      double dz = tower_armor_hs[i].second - tower_armor_hs[0].second;
      int dz_px = dz > 0 ? 1 : -1;
      int dz_mu;
      if (std::abs(dz) > TOWER_ARMOR_DTB) {
        dz_mu = 2;
      } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
        dz_mu = 1;
      } else {
        dz_mu = 0;
      }
      plate_z = center_fused.z() + ekf_.x[10] * dz_px * dz_mu;
    } else {
      plate_z = (use_l_h) ? center_fused.z() + ekf_.x[10] : center_fused.z();
    }

    _armor_xyza_list.push_back({plate_x, plate_y, plate_z, angle});
  }
  return _armor_xyza_list;
}

// 检查滤波器半径是否发散
bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
  if (r_ok && l_ok) return false;
  return true;
}

// 判断当前目标是否收敛
bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }
  return is_converged_;
}

// 核心函数：根据 EKF 状态和 ID 推算该装甲板在世界坐标系下的理论位置 XYZ
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);

  double armor_z;
  if(name == ArmorName::outpost){
      double dz = tower_armor_hs[id].second - tower_armor_hs[0].second;
      int dz_px = dz > 0 ? 1 : -1;
      int dz_mu;
      
      // 使用定义的常量区分大跳变和小跳变阶梯
      if (std::abs(dz) > TOWER_ARMOR_DTB) {
        dz_mu = 2; // 相隔两个阶梯 (大跳变)
      } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
        dz_mu = 1; // 相隔一个阶梯 (小跳变)
      } else {
        dz_mu = 0; // 同一阶梯
      }
      
      // 结合滤波器的高度参数 x[10]
      armor_z = x[4] + x[10] * dz_px * dz_mu; 
    } else {
      armor_z = (use_l_h) ? x[4] + x[10] : x[4];
    }
  return {armor_x, armor_y, armor_z};
}

// 计算当前预测观测函数的雅可比矩阵
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);
  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  double dz_dh;
  if(this->name == ArmorName::outpost){
    double dz = tower_armor_hs[id].second - tower_armor_hs[0].second;
    int dz_px = dz > 0 ? 1 : -1;
    int dz_mu;
    
    // 使用定义的常量区分跳变阶梯
    if (std::abs(dz) > TOWER_ARMOR_DTB) {
      dz_mu = 2;
    } else if (std::abs(dz) > TOWER_ARMOR_XTB) {
      dz_mu = 1;
    } else {
      dz_mu = 0;
    }
    dz_dh = dz_mu * dz_px;
  }else{
    dz_dh = (use_l_h) ? 1.0 : 0.0;
  }
  
  // 11 维位置偏导雅可比矩阵
  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, 11);
  H_armor_xyza(0, 0) = 1; H_armor_xyza(0, 6) = dx_da; H_armor_xyza(0, 8) = dx_dr; H_armor_xyza(0, 9) = dx_dl;
  H_armor_xyza(1, 2) = 1; H_armor_xyza(1, 6) = dy_da; H_armor_xyza(1, 8) = dy_dr; H_armor_xyza(1, 9) = dy_dl;
  H_armor_xyza(2, 4) = 1; H_armor_xyza(2, 10) = dz_dh;
  H_armor_xyza(3, 6) = 1;

  // 将 XYZ 偏导转换到 YPD 球坐标系下
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);

  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };

  // 链式求导法 H_Final = H_ypda * H_xyza
  return H_armor_ypda * H_armor_xyza;
}

// 检查是否完成初始化
bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
