#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "MvCameraControl.h"
#include "io/camera/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class HikRobot : public CameraBase
{
public:
  /** @brief 构造海康相机 @param sn 相机序列号 @param exposure_us 曝光时间，单位 us @param gain 增益 @param vid_pid USB VID:PID @param flip 是否垂直翻转 @param mirror 是否水平镜像 */
  HikRobot(std::string sn, double exposure_us, double gain, const std::string & vid_pid, bool flip, bool mirror);
  /** @brief 停止线程并释放海康相机资源 */
  ~HikRobot() override;
  /** @brief 阻塞读取一帧图像 @param img 输出图像 @param timestamp 输出采集时间戳 */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  /** @brief 尝试读取一帧图像 @param img 输出图像 @param timestamp 输出采集时间戳 @return 成功读取时返回 true */
  bool try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  /** @brief 清空软件队列和 SDK 图像缓存 */
  void clear_camera_frame_buffer() override
  {
    queue_.clear();
    if (handle_ != nullptr) MV_CC_ClearImageBuffer(handle_);
  }

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };
  
  size_t nDeviceNum = 0;//当前设备数量
  double exposure_us_;
  double gain_;
  bool flip_ = false;// 垂直翻转
  bool mirror_ = false;// 水平镜像

  std::thread daemon_thread_;
  std::atomic<bool> daemon_quit_;

  void * handle_ = nullptr;
  std::thread capture_thread_;
  
  std::atomic<bool> capture_quit_;
  tools::ThreadSafeQueue<CameraData, true> queue_;

  int vid_, pid_;

  /** @brief 从设备列表选择匹配序列号的相机 @param pDeviceInfo 设备信息数组 @param sn 目标序列号 @param cameraIndex 输出设备索引 @return 找到设备时返回 true */
  bool ChoiceCamrea(MV_CC_DEVICE_INFO** pDeviceInfo, 
                  unsigned char* sn, 
                  size_t& cameraIndex);
  /** @brief 启动相机采集线程 */
  void capture_start();
  /** @brief 停止相机采集线程 */
  void capture_stop();

  /** @brief 设置 SDK 浮点参数 @param name 参数名 @param value 参数值 */
  void set_float_value(const std::string & name, double value);
  /** @brief 设置 SDK 枚举参数 @param name 参数名 @param value 参数值 */
  void set_enum_value(const std::string & name, unsigned int value);

  /** @brief 解析 USB VID:PID 配置 @param vid_pid VID:PID 字符串 */
  void set_vid_pid(const std::string & vid_pid);
  /** @brief 复位相机对应的 USB 设备 */
  void reset_usb() const;

  /** @brief 暂停相机采集 */
  void pause() override;
  /** @brief 恢复相机采集 */
  void resume() override;

  std::mutex pause_mutex_;
  std::condition_variable pause_cv_;
  
};

}  // namespace io

#endif  // IO__HIKROBOT_HPP
