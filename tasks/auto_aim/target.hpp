#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "kf_example/rv_from_fyt.hpp"
#include "tools/fft.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only
  Eigen::Vector3d xyz_in_world;
  // rvFromFYT 残差平方和，x y z yaw
  std::optional<Eigen::Vector4d> rv_residual = std::nullopt;

  /** @brief 构造空目标 */
  Target();
  /** @brief 由首次装甲板观测初始化目标 @param armor 首次观测装甲板 @param t 观测时间 @param radius 初始旋转半径 @param armor_num 装甲板数量 @param P0_dig 初始协方差对角元素 */
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  /** @brief 构造用于测试的简化目标 @param x 初始 X 坐标 @param vyaw 初始偏航角速度 @param radius 旋转半径 @param h 目标高度 */
  Target(double x, double vyaw, double radius, double h);

  /** @brief 将目标预测到指定时间 @param t 目标时间 */
  void predict(std::chrono::steady_clock::time_point t);
  /** @brief 按时间步长预测目标 @param dt 时间步长，单位 s @param u_xyz 三轴控制输入 */
  void predict(double dt, Eigen::VectorXd u_xyz = Eigen::VectorXd::Zero(3));
  /** @brief 使用控制输入将目标预测到指定时间 @param t 目标时间 @param u_xyz 三轴控制输入 */
  void predict(std::chrono::steady_clock::time_point t, Eigen::VectorXd u_xyz);
  /** @brief 使用装甲板观测更新目标滤波器 @param armor 新观测装甲板 */
  void update(const Armor & armor);

  /** @brief 获取滤波状态副本 @return EKF 状态向量 */
  Eigen::VectorXd ekf_x() const;
  /** @brief 获取滤波器只读引用 @return RV 扩展卡尔曼滤波器 */
  const RVfromFYT & ekf() const;
  /** @brief 计算目标全部装甲板的坐标与偏航角 @return xyza 列表 */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  /** @brief 获取最近装甲板的位置、偏航和距离 @return xyzad 向量 */
  Eigen::Matrix<double, 5, 1> get_recent_armor_xyzad() const;

  /** @brief 根据状态计算指定编号装甲板的位置 @param x 目标状态 @param id 装甲板编号 @return 装甲板世界坐标 */
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  /** @brief 判断滤波器是否发散 @return 发散时返回 true */
  bool diverged() const;

  /** @brief 更新并查询滤波器收敛状态 @return 收敛时返回 true */
  bool convergened();

  bool isinit = false;

  /** @brief 检查目标是否完成初始化 @return 已初始化时返回 true */
  bool checkinit();

  /** @brief 获取 EKF 状态估计 @return 状态向量副本 */
  inline Eigen::VectorXd getEKFXest() {
    return ekf_.x;
  }

  /** @brief 获取目标当前滤波时间 @return 时间戳 */
  inline std::chrono::steady_clock::time_point getTimePoint() {
    return t_;
  }

  //前哨站
  std::pair<bool, double> tower_armor_hs[3] = {std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0), std::pair<bool, double>(false, 0)};
  // double tower_armor_hs[3] = {0,0,0};  
  double tower_armor_h = 0.0;
  double tower_armor_hs_datas[3] = {0,0,0}; 
  double last_tower_armor_h[3] = {0,0,0};
  int tower_armor_hs_datas_ptr[3] = {0, 0, 0};

  //长短焦
  bool cam_is_short = true;
  
  int update_count_;

  std::optional<tools::Wave> wave_;

private:
  int armor_num_;
  int switch_count_;
  

  bool is_switch_, is_converged_;

  RVfromFYT ekf_;
  State2Est* est = &ekf_;

  std::chrono::steady_clock::time_point t_;

  /** @brief 将前哨站装甲板高度观测同步到目标模型 */
  void sync_tower_armor_heights();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
