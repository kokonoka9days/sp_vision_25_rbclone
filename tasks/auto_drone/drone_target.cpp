#include "drone_target.hpp"

namespace auto_drone
{

Target::Target(const Drone & drone)
{
  name = drone.name;
  color = drone.color;

  // 修改：适配动态矩阵 API
  Eigen::VectorXd x(6);
  x << drone.xyz_in_world[0], drone.xyz_in_world[1], drone.xyz_in_world[2], 0.0, 0.0, 0.0;
  
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6, 6);
  P.topLeftCorner<3, 3>() *= 0.1;   
  P.bottomRightCorner<3, 3>() *= 1.0; 

  ekf_ = tools::ExtendedKalmanFilter(x, P);
}

void Target::predict(double dt)
{
  // 状态转移矩阵 F
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;

  // 过程噪声协方差 Q
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
  Q.topLeftCorner<3, 3>() *= 0.01;  
  Q.bottomRightCorner<3, 3>() *= 0.1; 

  // 非线性转移方程 f(x)
  auto f = [dt](const Eigen::VectorXd & x) {
    Eigen::VectorXd x_new = x;
    x_new[0] += x[3] * dt; 
    x_new[1] += x[4] * dt; 
    x_new[2] += x[5] * dt; 
    return x_new;
  };

  ekf_.predict(F, Q, f);
}

void Target::update(const Drone & drone)
{
  // 观测矩阵 H
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
  H.leftCols<3>() = Eigen::MatrixXd::Identity(3, 3);

  // 观测噪声协方差 R
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
  R *= 0.05; 

  // 观测值 z
  Eigen::VectorXd z(3);
  z << drone.xyz_in_world[0], drone.xyz_in_world[1], drone.xyz_in_world[2];

  // 观测方程 h(x)
  auto h = [](const Eigen::VectorXd & x) {
    return x.head<3>(); 
  };

  ekf_.update(z, H, R, h);
}

Eigen::Vector3d Target::get_xyz() const
{
  return ekf_.x.head<3>();
}

Eigen::Vector3d Target::get_v() const
{
  return ekf_.x.tail<3>();
}

}  // namespace auto_drone