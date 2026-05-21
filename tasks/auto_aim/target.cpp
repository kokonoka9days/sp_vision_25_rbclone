#include "target.hpp"

#include <numeric>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

#include <gtsam/inference/Symbol.h>
#include "factors.hpp"

// 物理常量定义
constexpr double TOWER_ARMOR_DH = 0.10;  // 前哨站两个装甲板之间的标准高低差(m)
constexpr double TOWER_ARMOR_DTB = 0.16;  // 前哨装甲大跳变阈值(m)
constexpr double TOWER_ARMOR_XTB = 0.05;  // 前哨装甲小跳变阈值(m)

using gtsam::symbol_shorthand::X; // 装甲车中心位置: Point3 
using gtsam::symbol_shorthand::V; // 装甲车线速度: Vector3
using gtsam::symbol_shorthand::R; // 装甲车自转 Yaw: Rot2
using gtsam::symbol_shorthand::W; // 装甲车自转角速度: double
const gtsam::Key RADIUS_KEY = gtsam::Symbol('r', 0); // 装甲板旋转半径
const gtsam::Key DZ_KEY = gtsam::Symbol('z', 0);     // 前哨站高低差

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
  motion_state_(MotionState::TRANSLATION) // 默认初始状态为平移模型
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
: armor_num_(4),
  motion_state_(MotionState::TRANSLATION)
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
  if (use_fgo_) {
    predict_fgo(dt);
    return;
  }

  // ==== 以下为完整保留的原有 EKF 预测逻辑 ====
  double vyaw = std::abs(ekf_.x[7]);
  double v_linear = std::hypot(ekf_.x[1], ekf_.x[3]); // 计算XY方向合成线速度 
  
  // 状态机滞回阈值配置
  const double OMEGA_HIGH = 3;     // 进入旋转的角速度阈值 (rad/s)
  const double OMEGA_LOW = 1.5;    // 退出旋转的角速度阈值 (rad/s)
  const double V_HIGH = 0.6;       // 进入平移旋转的线速度阈值 (m/s)
  const double V_LOW = 0.3;        // 退出平移旋转的线速度阈值 (m/s)

  // ================= 运动状态转移逻辑 =================
  switch (motion_state_) {
    case MotionState::TRANSLATION:
      if (vyaw > OMEGA_HIGH) {
        if (v_linear > V_HIGH) motion_state_ = MotionState::TRANSLATION_ROTATION;
        else motion_state_ = MotionState::IN_PLACE_ROTATION;
      }
      break;

    case MotionState::IN_PLACE_ROTATION:
      if (vyaw < OMEGA_LOW) {
        motion_state_ = MotionState::TRANSLATION;
      } else if (v_linear > V_HIGH) {
        motion_state_ = MotionState::TRANSLATION_ROTATION;
      }
      break;

    case MotionState::TRANSLATION_ROTATION:
      if (vyaw < OMEGA_LOW) {
        motion_state_ = MotionState::TRANSLATION;
      } else if (v_linear < V_LOW) {
        motion_state_ = MotionState::IN_PLACE_ROTATION;
      }
      break;
  }
  
  // 11维基础转移矩阵 F (x = F*x)
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(11, 11);
  F(0, 1) = dt; F(2, 3) = dt; F(4, 5) = dt; F(6, 7) = dt;

  double v1, v2;
  if (name == ArmorName::outpost) {
    ekf_.x(10) = TOWER_ARMOR_DH; // 修复原代码中 this->ekf_x()(10) 返回临时变量的问题
    // 前哨站位置固定，收敛后极大限制平移噪声
    if (this->convergened()) {
        v1 = 0.1;  // 锁死 X, Y, Z 中心
    } else {
        v1 = 20;   // 允许前期寻找中心
    }
    v2 = 0.1;      // 允许自转速度存在微小波动
  } 
  else {
    // 根据状态机分配不同的噪声
    switch (motion_state_) {
      case MotionState::TRANSLATION:
        v1 = 100; v2 = 20; // 灵活平移
        break;
      case MotionState::IN_PLACE_ROTATION:
        v1 = 1;   v2 = 0.1; // 抑制平移漂移，紧跟自转
        break;
      case MotionState::TRANSLATION_ROTATION:
        v1 = 100; v2 = 400; // 高机动状态，全部放开
        break;
    }
  }

  // 构造过程噪声矩阵 Q
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
}

void Target::predict_fgo(double dt) 
{
  if (current_estimate_.empty()) return;
  
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values new_values;
  
  auto prev_id = frame_id_;
  frame_id_++; 

  // 获取上一帧的状态
  auto prev_pos = current_estimate_.at<gtsam::Point3>(X(prev_id));
  auto prev_vel = current_estimate_.at<gtsam::Vector3>(V(prev_id));
  auto prev_yaw = current_estimate_.at<gtsam::Rot2>(R(prev_id));
  auto prev_vyaw = current_estimate_.at<double>(W(prev_id));

  // 基于匀速模型预测初值
  gtsam::Point3 curr_pos = prev_pos + prev_vel * dt;
  gtsam::Rot2 curr_yaw = prev_yaw * gtsam::Rot2::fromAngle(prev_vyaw * dt);
  
  new_values.insert(X(frame_id_), curr_pos);
  new_values.insert(V(frame_id_), prev_vel);
  new_values.insert(R(frame_id_), curr_yaw);
  new_values.insert(W(frame_id_), prev_vyaw);

  // 添加运动约束因子 (需根据实际工况调节 Sigma 参数)
  auto trans_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.1, 0.1, 0.1).finished());
  auto vel_noise   = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1.0, 1.0, 1.0).finished());
  auto yaw_noise   = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << 0.1).finished());
  auto vyaw_noise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << 1.0).finished());

  graph.add(TranslationFactor(trans_noise, X(prev_id), V(prev_id), X(frame_id_), dt));
  graph.add(VelocityFactor(vel_noise, V(prev_id), V(frame_id_)));
  graph.add(YawFactor(yaw_noise, R(prev_id), W(prev_id), R(frame_id_), dt));
  graph.add(VyawFactor(vyaw_noise, W(prev_id), W(frame_id_)));

  isam2_->update(graph, new_values);
  current_estimate_ = isam2_->calculateEstimate();
}

void Target::update(const Armor & armor)
{
  int id = 0;

  if (this->name == ArmorName::outpost) {
    // 【策略 A：前哨站专用】
    // 纯几何匹配(距离+复合角度)
    auto min_angle_error = 1e10;
    const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

    if (!use_fgo_) ekf_.x(10) = TOWER_ARMOR_DH; 

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

    // 迟滞防抖动
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
        tower_armor_hs[last_id].first = true;
        tower_armor_hs[last_id].second = tower_armor_hs_datas[last_id] / tower_armor_hs_datas_ptr[last_id];
      }
    }
  } else {
    is_switch_ = false;
  }

  // 累加当前块装甲板的高度特征
  if(name == ArmorName::outpost){
    double a = 0.1; 
    tower_armor_h = a * armor.xyz_in_world[2] + (1 - a) * last_tower_armor_h[id];
    tower_armor_hs_datas[id] += tower_armor_h;
    last_tower_armor_h[id] = tower_armor_h;
    tower_armor_hs_datas_ptr[id]++;     

    if(tower_armor_hs_datas[id] > 10000){
      tower_armor_hs_datas[id] = (tower_armor_hs_datas[id] / tower_armor_hs_datas_ptr[id]) * 600;
      tower_armor_hs_datas_ptr[id] = 600;
    }
  }

  last_id = id;
  update_count_++;    
  xyz_in_world = armor.xyz_in_world;

  // ==== 路由：决定使用哪种后端进行状态更新 ====
  if (use_fgo_) {
    update_fgo(armor, id);
  } else {
    update_ypda(armor, id);
  }
}

void Target::update_fgo(const Armor & armor, int id) 
{
  if (current_estimate_.empty()) return;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values new_values; // 观测步不新增独立状态节点，但需要将当前装甲板的Pose注入作为被约束变量

  // 将云台结算出的装甲板世界坐标转为 GTSAM Pose3 变量
  gtsam::Rot3 rot = gtsam::Rot3::Ypr(armor.ypr_in_world[0], armor.ypr_in_world[1], armor.ypr_in_world[2]); 
  gtsam::Point3 trans(armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2]);
  gtsam::Pose3 armor_pose_world(rot, trans);

  // 为这一帧的 Armor Pose 分配 Key 并插入初值
  gtsam::Key pose_key = gtsam::Symbol('p', frame_id_);
  new_values.insert(pose_key, armor_pose_world);

  // 利用一个较松的先验因子固定 PNP 解算出的 Pose，或者交由底层的重投影误差约束
  auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05).finished());
  graph.addPrior(pose_key, armor_pose_world, pose_noise);

  // 这里的 T_camera_to_odom 我们传入 Identity，因为传入的已经是世界坐标系的 Pose
  Eigen::Isometry3d T_identity = Eigen::Isometry3d::Identity();
  auto obs_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(4) << 0.05, 0.05, 0.05, 0.05).finished());

  if (name == ArmorName::outpost) {
      graph.add(ArmorRadiusDZFactor(obs_noise, pose_key, RADIUS_KEY, DZ_KEY, R(frame_id_), X(frame_id_), T_identity, static_cast<ArmorIndex>(id), 0.1, 0.5, armor_num_));
  } else {
      graph.add(ArmorRadiusCenterZFactor(obs_noise, pose_key, RADIUS_KEY, R(frame_id_), X(frame_id_), T_identity, static_cast<ArmorIndex>(id), 0.1, 0.5));
  }

  isam2_->update(graph, new_values);
  current_estimate_ = isam2_->calculateEstimate();
}

void Target::update_ypda(const Armor & armor, int id)
{
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  auto r2_azimuth = 4e-3;
  auto r2_pitch = 4e-3;
  auto r2_angle = log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2;
  auto r2_d = log(std::abs(delta_angle) + 1) + 1;

  if (motion_state_ == MotionState::IN_PLACE_ROTATION) {
    r2_angle *= 3.0;  // 旋转时进一步增加对角度的不信任
}

  // // 前哨站换板瞬间，放宽距离和高度噪声信任度防跳变
  // if (name == ArmorName::outpost && is_switch_) {
  //     r2_pitch *= 100.0; 
  //     r2_d     *= 100.0; 
  // }
  
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
}

void Target::init_fgo(const Armor & armor, double radius) 
{
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  isam2_ = std::make_shared<gtsam::ISAM2>(params);

  gtsam::Values initial_values;
  initial_values.insert(X(0), gtsam::Point3(armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2]));
  initial_values.insert(V(0), gtsam::Vector3(0, 0, 0));
  initial_values.insert(R(0), gtsam::Rot2::fromAngle(armor.ypr_in_world[0]));
  initial_values.insert(W(0), 0.0);
  initial_values.insert(RADIUS_KEY, radius);
  initial_values.insert(DZ_KEY, (name == ArmorName::outpost) ? TOWER_ARMOR_DH : 0.0);

  // 对第0帧添加强先验，防止漂移
  gtsam::NonlinearFactorGraph graph;
  auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.1, 0.1, 0.1).finished());
  graph.addPrior(X(0), gtsam::Point3(armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2]), prior_noise);

  isam2_->update(graph, initial_values);
  current_estimate_ = isam2_->calculateEstimate();
  frame_id_ = 0;
}

// 获取 EKF 状态向量
Eigen::VectorXd Target::ekf_x() const 
{ 
  // 当开启了 FGO 并且内部已经有估计值时，从 GTSAM 图中提取
  if (use_fgo_ && !current_estimate_.empty()) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(11);
    
    auto pos = current_estimate_.at<gtsam::Point3>(X(frame_id_));
    auto vel = current_estimate_.at<gtsam::Vector3>(V(frame_id_));
    auto yaw = current_estimate_.at<gtsam::Rot2>(R(frame_id_));
    auto vyaw = current_estimate_.at<double>(W(frame_id_));
    
    double radius = 0.2;
    if (current_estimate_.exists(RADIUS_KEY)) {
      radius = current_estimate_.at<double>(RADIUS_KEY);
    }
    
    double dz = 0.0;
    if (current_estimate_.exists(DZ_KEY)) {
      dz = current_estimate_.at<double>(DZ_KEY);
    }

    // 重新封装为旧系统的格式：
    // [0]x, [1]vx, [2]y, [3]vy, [4]z, [5]vz, [6]yaw, [7]vyaw, [8]r, [9]r_, [10]z_
    x << pos.x(), vel.x(), 
         pos.y(), vel.y(), 
         pos.z(), vel.z(), 
         yaw.theta(), vyaw, 
         radius, 0.0, dz; 
         
    return x;
  }
  
  // 没有使用 FGO 或者数据还未初始化完成时，降级使用 EKF 的原生结构
  return ekf_.x; 
}

// 获取滤波器常引用
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

// 返回所有装甲板的预测四维状态 (X, Y, Z, Angle) 列表
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