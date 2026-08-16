#ifndef TOOLS__RECORDER_HPP
#define TOOLS__RECORDER_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "tools/thread_safe_queue.hpp"
namespace tools
{
class Recorder
{
public:
  /** @brief 构造异步视频记录器 @param fps 输出视频帧率 */
  Recorder(double fps = 30);
  /** @brief 停止保存线程并关闭输出文件 */
  ~Recorder();
  /** @brief 将一帧数据加入异步保存队列 @param img 图像 @param q 拍摄时云台姿态四元数 @param timestamp 拍摄时间戳 @param stream_id 视频流标识 */
  void record(
    const cv::Mat & img, const Eigen::Quaterniond & q,
    const std::chrono::steady_clock::time_point & timestamp,
    const std::string & stream_id = "default");

private:
  struct FrameData
  {
    cv::Mat img;
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
    std::string stream_id;
  };
  bool init_;
  std::atomic<bool> stop_thread_;
  double fps_;
  std::string text_path_;
  std::string video_base_path_;
  std::string video_path_;
  std::string active_stream_id_;
  std::string last_stream_id_;
  std::ofstream text_writer_;
  cv::VideoWriter video_writer_;
  cv::Size video_size_;
  int video_type_ = -1;
  std::size_t segment_index_ = 0;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_time_;
  tools::ThreadSafeQueue<FrameData, true> queue_;
  std::thread saving_thread_;  // 负责保存帧数据的线程
  /** @brief 使用首帧信息初始化输出文件 @param img 首帧图像 @param stream_id 视频流标识 */
  void init(const cv::Mat & img, const std::string & stream_id);
  /** @brief 打开新的视频分段 @param img 用于确定尺寸和类型的图像 @param stream_id 视频流标识 @return 打开成功时返回 true */
  bool open_video_segment(const cv::Mat & img, const std::string & stream_id);
  /** @brief 保存线程入口，持续消费队列并写入文件 */
  void save_to_file();
};

}  // namespace tools

#endif  // TOOLS__RECORDER_HPP
