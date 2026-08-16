#ifndef TOOLS__PLOTTER_HPP
#define TOOLS__PLOTTER_HPP

#include <netinet/in.h>  // sockaddr_in

#include <chrono>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

namespace tools
{
class Plotter
{
public:
  /**
   * @brief 构造 UDP 绘图数据发送器
   * @param host 目标主机地址
   * @param port 目标 UDP 端口
   * @param record 是否从构造时开始保存
   * @param record_path CSV 保存路径；为空时自动生成
   */
  Plotter(
    std::string host = "127.0.0.1", uint16_t port = 9870, bool record = true,
    std::string record_path = "");

  /** @brief 关闭网络套接字 */
  ~Plotter();

  /** @brief 将 JSON 数据发送给绘图端 @param json 待发送的数据 */
  void plot(const nlohmann::json & json);

  /**
   * @brief 开始保存后续绘图数据，停止时生成 PlotJuggler CSV
   * @param path CSV 或原始 JSONL 路径；为空时自动保存到 plot_records/时间戳.csv
   * @return 原始记录文件成功打开时返回 true
   */
  bool start_recording(const std::string & path = "");

  /** @brief 停止保存并生成 PlotJuggler 可直接读取的 CSV @return CSV 生成成功时返回 true */
  bool stop_recording();

  /** @brief 查询是否正在保存数据 */
  bool recording() const;

  /** @brief 获取当前或最近一次使用的记录文件路径 */
  std::string recording_path() const;

  /**
   * @brief 按记录的时间间隔将文件重新发送给 PlotJuggler
   * @param path start_recording 生成的 CSV 或 JSONL 文件
   * @param playback_speed 回放倍速，必须大于 0
   * @return 文件完整读取且所有数据成功发送时返回 true
   */
  bool replay(const std::string & path, double playback_speed = 1.0);

private:
  /** @brief 在已经持有 mutex_ 时发送数据 */
  bool send_unlocked(const nlohmann::json & json);

  /** @brief 关闭原始记录并生成 CSV；调用前必须持有 mutex_ */
  bool finish_recording_unlocked();

  int socket_ = -1;
  sockaddr_in destination_{};
  mutable std::mutex mutex_;
  std::ofstream record_stream_;
  std::string recording_path_;
  std::string raw_recording_path_;
  bool recording_pending_ = false;
  std::chrono::steady_clock::time_point recording_start_;
};

}  // namespace tools

#endif  // TOOLS__PLOTTER_HPP
