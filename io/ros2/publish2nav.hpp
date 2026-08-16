#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <Eigen/Dense>  // For Eigen::Vector3d
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace io
{
class Publish2Nav : public rclcpp::Node
{
public:
  /** @brief 创建导航目标 ROS2 发布节点 */
  Publish2Nav();

  /** @brief 销毁发布节点 */
  ~Publish2Nav();

  /** @brief 初始化目标话题发布者 */
  void start();

  /** @brief 发布目标位置 @param data 目标的四维位置数据 */
  void send_data(const Eigen::Vector4d & data);

private:
  // ROS2 发布者
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
