#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(std::string sn, double exposure_us, double gain, const std::string & vid_pid, bool flip, bool mirror)
: camera_sn_(sn), exposure_us_(exposure_us), gain_(gain), queue_(1), daemon_quit_(false), vid_(-1), pid_(-1), flip_(flip), mirror_(mirror)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

bool HikRobot::try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  bool read_full =  queue_.try_pop(data);

  
  if(read_full) {
    img = data.img;
    timestamp = data.timestamp;
    last_read_t = data.timestamp;
  }  
  return read_full;
}


bool HikRobot::ChoiceCamrea(MV_CC_DEVICE_INFO** pDeviceInfo, unsigned char* sn, size_t& cameraIndex){
    for(size_t i = 0; i < nDeviceNum; i++){
        std::cout<<"pDeviceInfo "<<i<<": "<<pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber<<std::endl;
        // if(*pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber == *sn) {
        //     nDeviceNum = i;
        //     return true;
        // }
        bool wl = true;
        for(int j  = 0; sn[j]!='\0';j++){
            if(sn[j] != pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber[j]) {
                wl = false;
                break;
            }
        }
        if(wl) {
            cameraIndex = i;
            return true;
        }
    }
    return false;

}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;
  MV_CC_DEVICE_INFO_LIST device_list;

  // 1. 枚举设备
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if(ret != MV_OK) {
      tools::logger()->warn("hik EnumDevices failed, ret: {:#x}", ret);
      return;
  }
  if(device_list.nDeviceNum == 0) {
      tools::logger()->warn("设备数量为0");
      return;
  }
  this->nDeviceNum = device_list.nDeviceNum;

  // 2. 匹配目标相机
  size_t cameraIndex = 0;
  bool exist = ChoiceCamrea(device_list.pDeviceInfo, (unsigned char*)camera_sn_.c_str(), cameraIndex);
  if(!exist){
    tools::logger()->warn("未匹配到指定SN的海康相机: {}", camera_sn_);
    return;
  }

  // 3. 创建句柄
  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[cameraIndex]);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    return;
  }

  // 4. 打开设备
  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    return;
  }

  // 5. 核心参数设置 (极其重要)
  set_enum_value("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS); // 强制连续采集模式
  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_enum_value("TriggerMode", MV_TRIGGER_MODE_OFF);

  // 6. 安全地设置曝光
  set_float_value("ExposureTime", exposure_us_);

  // 7. 安全地设置增益（修复垃圾值 BUG）
  MVCC_FLOATVALUE gainRange = {0}; // 【务必初始化为 0】
  ret = MV_CC_GetFloatValue(handle_, "Gain", &gainRange); // 节点名改为 "Gain"
  if (ret == MV_OK) {
      double target_gain = gain_ * gainRange.fMax;
      // 钳制在合法范围内
      if (target_gain < gainRange.fMin) target_gain = gainRange.fMin;
      if (target_gain > gainRange.fMax) target_gain = gainRange.fMax;
      set_float_value("Gain", target_gain);
  } else {
      tools::logger()->warn("获取增益范围失败: {:#x}，跳过增益设置", ret);
  }

  // 8. 限制帧率（保命设置：防止 USB 带宽被打爆导致全损丢包）
  MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable",false);
  // set_float_value("AcquisitionFrameRate", 20.0f); // 强制降频到20帧，跑通后再往上加

  // 9. 开始取流
  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  // 10. 启动抓图线程
  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    capturing_ = true;
    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;
    memset(&cvt_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));

    // 修复点：在 lambda 内部显式声明局部变量 ret
    unsigned int ret; 

    while (!capture_quit_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      if (is_paused_) {
          std::unique_lock<std::mutex> lock(pause_mutex_);
          pause_cv_.wait(lock, [this]() { return !is_paused_.load(); });
      }

      // 获取图像，超时设为 2000ms
      ret = MV_CC_GetImageBuffer(handle_, &raw, 2000);
      if (ret != MV_OK) {
        if (is_paused_) continue; 
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x} 海康相机无法读取到图像", ret);
        
        // 【关键修复】：丢包或超时是正常的，继续等下一帧，不要 break 杀死线程！
        continue; 
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
        
      // 【防崩溃保护】检查像素格式是否在 type_map 中
      if (type_map.find(pixel_type) != type_map.end()) {
          cv::cvtColor(img, dst_image, type_map.at(pixel_type));
          img = dst_image;
      } else {
          img = img.clone(); 
      }

      // 翻转和镜像
      if (flip_) cv::flip(img, img, 0); 
      if (mirror_) cv::flip(img, img, 1); 

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        // 这里 free 失败通常意味着设备句柄无效了，此时 break 是合理的
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

void HikRobot::pause() {
    this->is_paused_ = true; // 设置暂停标志位
    if (handle_ != nullptr) {
        MV_CC_StopGrabbing(handle_);
    }
}

void HikRobot::resume() {
    this->is_paused_ = false; // 清除暂停标志位
    if (handle_ != nullptr) {
        MV_CC_StartGrabbing(handle_);
    }
    pause_cv_.notify_all(); // 唤醒正在沉睡的线程
}

}  // namespace io