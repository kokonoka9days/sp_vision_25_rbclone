#ifndef IO__ROS2_HPP
#define IO__ROS2_HPP

#include "publish2nav.hpp"
#include "subscribe2nav.hpp"

namespace io
{
class ROS2
{
public:
  /** @brief 初始化 ROS2 发布与订阅节点 */
  ROS2();

  /** @brief 停止 ROS2 节点线程 */
  ~ROS2();

  /** @brief 发布目标位置 @param target_pos 目标四维位置 */
  void publish(const Eigen::Vector4d & target_pos);

  /** @brief 获取最新敌方状态 @return 敌方状态数组 */
  std::vector<int8_t> subscribe_enemy_status();

  /** @brief 获取最新自瞄目标 @return 自瞄目标数组 */
  std::vector<int8_t> subscribe_autoaim_target();

  /** @brief 创建发布者并在独立线程中运行节点 @tparam T ROS2 消息类型 @param node_name 节点名 @param topic_name 话题名 @param queue_size 发布队列长度 @return 发布者共享指针 */
  template <typename T>
  std::shared_ptr<rclcpp::Publisher<T>> create_publisher(
    const std::string & node_name, const std::string & topic_name, size_t queue_size)
  {
    auto node = std::make_shared<rclcpp::Node>(node_name);

    auto publisher = node->create_publisher<T>(topic_name, queue_size);

    // 运行一个单独的线程来 spin 这个节点，确保消息可以被正确发布
    std::thread([node]() { rclcpp::spin(node); }).detach();

    return publisher;
  }

private:
  std::shared_ptr<Publish2Nav> publish2nav_;
  std::shared_ptr<Subscribe2Nav> subscribe2nav_;

  std::unique_ptr<std::thread> publish_spin_thread_;
  std::unique_ptr<std::thread> subscribe_spin_thread_;
};

}  // namespace io
#endif
