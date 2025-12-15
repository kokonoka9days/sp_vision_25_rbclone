#ifndef DAHENG_CAMERA_HPP
#define DAHENG_CAMERA_HPP

#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

#include "GxIAPI.h"  // 大恒相机SDK头文件
#include "tools/thread_safe_queue.hpp"
#include "io/camera.hpp"
#include "tools/logger.hpp"

namespace io
{


class DahengCamera  : public CameraBase{
public:
    /**
     * @brief 构造函数
     * @param exposure_ms 曝光时间(毫秒)
     * @param gain 增益值
     * @param frame_rate 帧率
     * @param serial_number 相机序列号(为空时使用第一个相机)
     */
    DahengCamera::DahengCamera(std::string camera_sn, 
                            double exposure_us, 
                            double gain, 
                            double gamma,
                           double frame_rate);
    
    ~DahengCamera();


    bool capture_stop();
    
    /**
     * @brief 读取图像
     * @param img 输出的图像
     * @param timestamp 时间戳
     */
    void read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp);
    
    /**
     * @brief 检查相机是否连接
     */
    bool is_connected() const { return connected_; }
    
    /**
     * @brief 设置曝光时间
     * @param exposure_ms 曝光时间(毫秒)
     */
    void set_exposure(double exposure_ms);
    
    /**
     * @brief 设置增益
     * @param gain 增益值
     */
    void set_gain(double gain);
    
    /**
     * @brief 设置帧率
     * @param frame_rate 帧率
     */
    void set_frame_rate(double frame_rate);
    
    /**
     * @brief 设置触发模式
     * @param trigger_mode true: 外部触发, false: 连续采集
     */
    void set_trigger_mode(bool trigger_mode);
    
    /**
     * @brief 设置白平衡
     * @param r_gain 红色通道增益
     * @param g_gain 绿色通道增益
     * @param b_gain 蓝色通道增益
     */
    void set_white_balance(double r_gain, double g_gain, double b_gain);

private:

    struct CameraData {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;
    };
    // 相机操作
    bool initialize_camera();
    bool open_camera();
    bool close_camera();
    bool start_acquisition();
    bool stop_acquisition();
    bool configure_camera();
    void reset_camera();
    
    // 线程函数
    void capture_thread_func();
    void daemon_thread_func();
    
    // 参数设置辅助函数
    bool set_float_value(const char* feature_name, double value);
    bool set_int_value(const char* feature_name, int64_t value);
    bool set_enum_value(const char* feature_name, const char* value);
    
    // 图像处理
    cv::Mat convert_raw_to_mat(void* raw_data, size_t width, size_t height, 
                               GX_PIXEL_FORMAT_ENTRY pixel_format);

private:
    // 相机参数
    std::string camera_sn_;
    double exposure_us_;
    double gain_;
    double gamma_;
    double frame_rate_;
    
    // 相机句柄和状态
    GX_DEV_HANDLE hDevice = nullptr;
    GX_OPEN_PARAM* open_param_ = nullptr;
    int64_t PixelFormat = GX_PIXEL_FORMAT_BAYER_GR8;
    int64_t ColorFilter = GX_COLOR_FILTER_NONE;
    GX_FRAME_DATA frameData{};
    void *pRaw8Buffer = nullptr;
    void *pMirrorBuffer = nullptr;
    void *pRGBframeData = nullptr;
    void *pGammaLut = nullptr;
    
    std::atomic<bool> connected_{false};
    std::atomic<bool> capturing_{false};
    
    // 线程控制
    std::atomic<bool> daemon_quit_{false};
    std::atomic<bool> capture_quit_{false};
    std::thread daemon_thread_;
    std::thread capture_thread_;
    
    // 数据队列
    tools::ThreadSafeQueue<CameraData> queue_;
    
    // SDK状态
    bool sdk_initialized_ = false;
    
    // 图像参数缓存
    size_t image_width_ = 0;
    size_t image_height_ = 0;
    // GX_PIXEL_FORMAT_ENTRY pixel_format_ = GX_PIXEL_FORMAT_MONO8;
    
    // 配置参数
    bool trigger_mode_ = false;
    bool auto_white_balance_ = true;
};

}  // namespace io

#endif  // DAHENG_CAMERA_HPP