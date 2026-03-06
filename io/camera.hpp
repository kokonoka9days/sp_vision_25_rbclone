#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace io
{
class CameraBase
{
public:
  int64_t sensorWidth = -1, sensorHeight = -1; //相机分辨率
  std::atomic<bool> is_paused_{false};

  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;

  // 新增虚函数接口
  virtual void pause() {} //停止
  virtual void resume() {} //开启
};

class Camera
{
public:
  std::string main_and_secondary = "main"; //是否是主相机
  std::chrono::microseconds timestamp_offset = std::chrono::microseconds(0); //时间戳偏移量

  Camera(const std::string & config_path);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  // 新增对外调用的接口
  void pause() { if(camera_) camera_->pause(); }
  void resume() { if(camera_) camera_->resume(); }
  bool is_paused()  {   
    if(camera_) return camera_->is_paused_;
    else return false;
    }
  

private:
  std::unique_ptr<CameraBase> camera_;
};

}  // namespace io

#endif  // IO__CAMERA_HPP