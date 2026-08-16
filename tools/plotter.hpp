#ifndef TOOLS__PLOTTER_HPP
#define TOOLS__PLOTTER_HPP

#include <netinet/in.h>  // sockaddr_in

#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

namespace tools
{
class Plotter
{
public:
  /** @brief 构造 UDP 绘图数据发送器 @param host 目标主机地址 @param port 目标 UDP 端口 */
  Plotter(std::string host = "127.0.0.1", uint16_t port = 9870);

  /** @brief 关闭网络套接字 */
  ~Plotter();

  /** @brief 将 JSON 数据发送给绘图端 @param json 待发送的数据 */
  void plot(const nlohmann::json & json);

private:
  int socket_;
  sockaddr_in destination_;
  std::mutex mutex_;
};

}  // namespace tools

#endif  // TOOLS__PLOTTER_HPP
