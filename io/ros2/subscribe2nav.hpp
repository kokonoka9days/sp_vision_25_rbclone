#ifndef IO__SUBSCRIBE2NAV_HPP
#define IO__SUBSCRIBE2NAV_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include "sp_msgs/msg/autoaim_target_msg.hpp"
#include "sp_msgs/msg/enemy_status_msg.hpp"
#include <vector>
#include "tools/thread_safe_queue.hpp"

namespace io
{
class Subscribe2Nav : public rclcpp::Node
{
public:
  /** @brief 创建导航消息 ROS2 订阅节点 */
  Subscribe2Nav();

  /** @brief 销毁订阅节点 */
  ~Subscribe2Nav();

  /** @brief 初始化导航相关话题订阅 */
  void start();

  /** @brief 获取最新敌方状态 @return 敌方状态数组 */
  std::vector<int8_t> subscribe_enemy_status();
  /** @brief 获取最新自瞄目标信息 @return 自瞄目标数组 */
  std::vector<int8_t> subscribe_autoaim_target();

private:
  /** @brief 处理敌方状态消息 @param msg ROS2 消息指针 */
  void enemy_status_callback(const sp_msgs::msg::EnemyStatusMsg::SharedPtr msg);
  /** @brief 处理自瞄目标消息 @param msg ROS2 消息指针 */
  void autoaim_target_callback(const sp_msgs::msg::AutoaimTargetMsg::SharedPtr msg);

  int enemy_status_counter_;
  int autoaim_target_counter_;

  rclcpp::TimerBase::SharedPtr enemy_status_timer_;
  rclcpp::TimerBase::SharedPtr autoaim_target_timer_;

  rclcpp::Subscription<sp_msgs::msg::EnemyStatusMsg>::SharedPtr enemy_status_subscription_;
  rclcpp::Subscription<sp_msgs::msg::AutoaimTargetMsg>::SharedPtr autoaim_target_subscription_;

  tools::ThreadSafeQueue<sp_msgs::msg::EnemyStatusMsg> enemy_statue_queue_;
  tools::ThreadSafeQueue<sp_msgs::msg::AutoaimTargetMsg> autoaim_target_queue_;
};
}  // namespace io

#endif  // IO__SUBSCRIBE2NAV_HPP
