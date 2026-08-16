#ifndef IO__MINDVISION_HPP
#define IO__MINDVISION_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "CameraApi.h"
#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class MindVision : public CameraBase
{
public:
  /** @brief 构造迈德威视相机 @param exposure_us 曝光时间，单位 us @param gamma 伽马值 @param vid_pid USB VID:PID */
  MindVision(double exposure_us, double gamma, const std::string & vid_pid);
  /** @brief 停止线程并释放相机资源 */
  ~MindVision() override;
  /** @brief 阻塞读取一帧图像 @param img 输出图像 @param timestamp 输出采集时间戳 */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  /** @brief 尝试读取一帧图像 @param img 输出图像 @param timestamp 输出采集时间戳 @return 成功读取时返回 true */
  bool try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  /** @brief 清空采集帧队列 */
  void clear_camera_frame_buffer() override { queue_.clear(); }
private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_us_, gamma_;
  CameraHandle handle_;
  int height_, width_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData, true> queue_;
  int vid_, pid_;

  /** @brief 打开并配置相机 */
  void open();
  /** @brief 尝试打开相机，失败时保持守护线程运行 */
  void try_open();
  /** @brief 关闭相机 */
  void close();
  /** @brief 解析 USB VID:PID 配置 @param vid_pid VID:PID 字符串 */
  void set_vid_pid(const std::string & vid_pid);
  /** @brief 复位相机对应的 USB 设备 */
  void reset_usb() const;

   
};

}  // namespace io

#endif  // IO__MINDVISION_HPP
