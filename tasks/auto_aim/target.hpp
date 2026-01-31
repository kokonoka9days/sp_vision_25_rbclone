#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

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

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  inline Eigen::VectorXd getEKFXest(){
    return ekf_.x;
  }

  inline std::chrono::steady_clock::time_point getTimePoint(){
    return t_;
  }

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;


  //前哨站
  // 从低到高逆时针，第一个最低，逆时针为正方向
  double last_last_tower_armor_h = 0, 
          last_tower_armor_h = 0, 
          now_tower_armor_h = 0;
  double tower_armor_hs[3] = {0,0,0};
  int  current_tower_armor = 1; //默认指向第二个
  bool tower_initialized = false;//前哨站建模完成

  /// @brief 获取前哨站角速度方向
  /// @return true表示角速度方向为正
  inline bool getTowerWAnglePositive(){
      return this->ekf_.x(7) > 0;
  }

  /// @brief 获取下一个前哨站装甲板编号
  /// @return 下一个前哨站装甲板编号
  inline int getNextCurrentTowerArmor(){
      return getTowerWAnglePositive() ? 
          (current_tower_armor + 1) % 3 : (current_tower_armor + 2) % 3;
  }

  /// @brief 返回当前观察前哨站装甲板编号
  /// @return 
  inline int getNowCurrentTowerArmor(){
      return current_tower_armor;
  }

  inline void UpdateNowArmorH(double now_tower_armor_h_){
      now_tower_armor_h = now_tower_armor_h_;
  }

  /// @brief 更新前哨站装甲板高度信息
  /// @param now_tower_armor_h_ 
  inline void UpdateTowerArmor(double now_tower_armor_h_){
      last_last_tower_armor_h = last_tower_armor_h;
      last_tower_armor_h = now_tower_armor_h;
      now_tower_armor_h = now_tower_armor_h_;
      // current_tower_armor = getNextCurrentTowerArmor();
  }
  /// @brief 返回前哨站其他装甲板落点差，逆时针为正方向
  /// @param cur_id 输入装甲板id号
  /// @param next_num 下几块装甲板
  /// @return 
  inline int getTowerOtherArmorGap(size_t cur_id, int next_num){
    --next_num;
    return -cur_id + ((cur_id + next_num + 1) % 3);
  }
  
  /// @brief 前哨站解算
  void SolveTower();

};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP