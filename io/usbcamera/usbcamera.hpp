#ifndef IO__USBCamera_HPP
#define IO__USBCamera_HPP

#include <chrono>
#include <atomic>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace io
{
class USBCamera
{
public:
  /** @brief 根据设备名和配置创建 USB 相机 @param open_name 设备路径或名称 @param config_path YAML 配置文件路径 */
  USBCamera(const std::string & open_name, const std::string & config_path);
  /** @brief 停止采集线程并关闭设备 */
  ~USBCamera();
  /** @brief 阻塞读取一帧图像 @return 图像 */
  cv::Mat read();
  /** @brief 阻塞读取一帧图像及时间戳 @param img 输出图像 @param timestamp 输出采集时间戳 */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  /** @brief 在限定时间内读取一帧 @param img 输出图像 @param timestamp 输出采集时间戳 @param timeout 最长等待时间 @return 成功读取时返回 true */
  bool read_for(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout);
  std::string device_name;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::mutex cap_mutex_;
  cv::VideoCapture cap_;
  cv::Mat img_;
  std::string open_name_;
  int usb_exposure_, usb_frame_rate_, sharpness_;
  int open_count_;
  double image_width_, image_height_;
  int usb_gamma_, usb_gain_;
  std::atomic_bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData> queue_;

  /** @brief 尝试打开相机，失败时记录状态 */
  void try_open();
  /** @brief 打开相机并启动采集 */
  void open();
  /** @brief 关闭相机设备 */
  void close();
};

}  // namespace io

#endif
