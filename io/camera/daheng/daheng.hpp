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
#include "DxImageProc.h"

#include "tools/thread_safe_queue.hpp"
#include "io/camera.hpp"
#include "tools/logger.hpp"

namespace io
{


class DahengCamera  : public CameraBase{
public:

    /** @brief 初始化大恒相机 SDK */
    static void initSDK(){
        // 初始化SDK
        GX_STATUS status = GXInitLib();
        if (status != GX_STATUS_SUCCESS) {
            tools::logger()->error("[Daheng] 大恒相机初始化失败，错误码: {:#x}", status);
            return;
        }
    }
    /**
     * @brief 构造函数
     * @param camera_sn 相机序列号，为空时使用首个设备
     * @param exposure_us 曝光时间，单位 us
     * @param gain 增益值
     * @param gamma 伽马值
     * @param flip 是否垂直翻转
     * @param mirror 是否水平镜像
     */
    DahengCamera(std::string camera_sn, 
                                double exposure_us, 
                                double gain, 
                                double gamma,
                                bool flip,
                                bool mirror
                            );
    
    /** @brief 停止采集并释放大恒相机资源 */
    ~DahengCamera();

    /** @brief 停止图像采集 @return 停止成功时返回 true */
    bool capture_stop();
    
    /**
     * @brief 读取图像
     * @param img 输出的图像
     * @param timestamp 时间戳
     */
    void read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp) override;
    /** @brief 尝试读取一帧图像 @param img 输出图像 @param timestamp 输出采集时间戳 @return 成功读取时返回 true */
    bool try_read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
    /** @brief 清空采集帧队列 */
    void clear_camera_frame_buffer() override { queue_.clear(); }
private:

    struct CameraData {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;
    };
    /** @brief 枚举并确认配置相机存在 @return 找到相机时返回 true */
    bool enum_and_check_camera();  // 枚举并检查相机
    /** @brief 配置相机采集参数 @return 初始化成功时返回 true */
    bool initialize_camera();
    /** @brief 打开相机并启动采集 @return 打开成功时返回 true */
    bool open_camera();
    /** @brief 关闭相机设备 @return 关闭成功时返回 true */
    bool close_camera();

    /** @brief 从 SDK 获取并转换一帧图像 @return 转换后的 OpenCV 图像 */
    cv::Mat getFrame( );
    /** @brief 转换并修正相机原始图像 @param pImageBuf SDK 输入图像 @param pImageRaw8Buf 8 位原始缓冲区 @param pImageRGBBuf RGB 输出缓冲区 @param nImageWidth 图像宽度 @param nImageHeight 图像高度 @param nPixelFormat 像素格式 @param nPixelColorFilter Bayer 滤色器类型 @param flip 是否垂直翻转 @param mirror 是否水平镜像 */
    void ProcessData(void *pImageBuf, void *pImageRaw8Buf, void *pImageRGBBuf, int nImageWidth, int nImageHeight,
                        int nPixelFormat, int nPixelColorFilter, bool flip , bool mirror ) ;
private:

    // 相机参数
    // std::string camera_sn_;
    double exposure_us_;
    double gain_;
    double gamma_;
    double frame_rate_;
    std::string open_content_; 
    bool flip_ = false;// 垂直翻转
    bool mirror_ = false;// 水平镜像
    
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
    
    // std::atomic<bool> connected_{false};
    // std::atomic<bool> capturing_{false};
    
    // 线程控制
    std::atomic<bool> daemon_quit_{false};
    std::atomic<bool> capture_quit_{false};
    // std::atomic<bool> is_stop_collecting{false};// 断采集
    std::thread daemon_thread_;
    std::thread capture_thread_;
    size_t stop_collecting_num = 0;
    
    // 数据队列
    tools::ThreadSafeQueue<CameraData, true> queue_;
    
    // SDK状态
    bool sdk_initialized_ ;//= false;
    
    // 图像参数缓存
    size_t image_width_ = 0;
    size_t image_height_ = 0;
    // GX_PIXEL_FORMAT_ENTRY pixel_format_ = GX_PIXEL_FORMAT_MONO8;
    
    // 配置参数
    bool trigger_mode_ = false;
    bool auto_white_balance_ = true;
    

    /** @brief 暂停相机采集 */
    void pause() override;
    /** @brief 恢复相机采集 */
    void resume() override;

    std::mutex pause_mutex_;
    std::condition_variable pause_cv_;
};

}  // namespace io

#endif  // DAHENG_CAMERA_HPP
