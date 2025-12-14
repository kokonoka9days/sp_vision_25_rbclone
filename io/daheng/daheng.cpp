#include "daheng.hpp"
#include <chrono>
#include <functional>
#include <vector>

namespace io
{

// SDK错误处理宏
#define DAHENG_CHECK(status) \
    if (status != GX_STATUS_SUCCESS) { \
        tools::logger()->error("Daheng Camera Error [{}]: {}", status, #status); \
        return false; \
    }

DahengCamera::DahengCamera(double exposure_ms, double gain, 
                           double frame_rate, const std::string& serial_number)
    : exposure_us_(exposure_ms * 1000.0), 
      gain_(gain), 
      frame_rate_(frame_rate), 
      serial_number_(serial_number),
      queue_(1)
{
    tools::logger()->info("Initializing Daheng Camera...");
    
    // 初始化SDK
    GX_STATUS status = GXInitLib();
    if (status != GX_STATUS_SUCCESS) {
        tools::logger()->error("Failed to initialize Daheng SDK: {:#x}", status);
        return;
    }
    sdk_initialized_ = true;
    
    // 启动守护线程
    daemon_thread_ = std::thread([&](){
        tools::logger()->info("Daemon thread started.");
        
        // 初始连接相机
        if (open_camera()) {
            start_acquisition();
        }
        
        while (!daemon_quit_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            if (capturing_) {
                continue;  // 正常采集中
            }
            
            // 相机断开，尝试重连
            tools::logger()->warn("Camera disconnected, attempting to reconnect...");
            
            capture_stop();
            reset_camera();
            
            // 等待一段时间再检查
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        // 清理
        capture_stop();
        close_camera();
        
        tools::logger()->info("Daemon thread stopped.");
    });
    
    tools::logger()->info("Daheng Camera initialized successfully.");
}

DahengCamera::~DahengCamera()
{
    tools::logger()->info("Daheng Camera destructor called.");
    
    // 停止守护线程
    daemon_quit_ = true;
    if (daemon_thread_.joinable()) {
        daemon_thread_.join();
        tools::logger()->debug("Daemon thread joined.");
    }
    
    // 关闭相机
    if (connected_) {
        stop_acquisition();
        close_camera();
    }
    
    // 释放SDK资源
    if (sdk_initialized_) {
        GX_STATUS status = GXCloseLib();
        if (status != GX_STATUS_SUCCESS) {
            tools::logger()->warn("Failed to close Daheng SDK: {:#x}", status);
        }
        sdk_initialized_ = false;
    }
    
    tools::logger()->info("Daheng Camera destroyed.");
}

void DahengCamera::read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp)
{
    CameraData data;
    queue_.pop(data);
    img = data.img;
    timestamp = data.timestamp;
}

bool DahengCamera::initialize_camera()
{
    if (!sdk_initialized_) {
        tools::logger()->error("SDK not initialized!");
        return false;
    }
    
    // 枚举设备
    uint32_t device_num = 0;
    GX_STATUS status = GXUpdateDeviceList(&device_num, 1000);
    DAHENG_CHECK(status);
    
    if (device_num == 0) {
        tools::logger()->warn("No Daheng camera found!");
        return false;
    }
    
    // 获取设备信息列表
    GX_DEVICE_BASE_INFO* device_info_list = new GX_DEVICE_BASE_INFO[device_num];
    status = GXGetAllDeviceBaseInfo(device_info_list, &device_num);
    if (status != GX_STATUS_SUCCESS) {
        delete[] device_info_list;
        DAHENG_CHECK(status);
    }
    
    // 选择设备
    int selected_index = 0;
    if (!serial_number_.empty()) {
        for (uint32_t i = 0; i < device_num; i++) {
            if (std::string(device_info_list[i].szSN) == serial_number_) {
                selected_index = i;
                break;
            }
        }
    }
    
    // 打开设备
    status = GXOpenDeviceByIndex(selected_index + 1, &device_handle_);
    delete[] device_info_list;
    DAHENG_CHECK(status);
    
    connected_ = true;
    tools::logger()->info("Daheng camera opened successfully.");
    
    // 配置相机参数
    return configure_camera();
}

bool DahengCamera::open_camera()
{
    if (connected_) {
        tools::logger()->warn("Camera already opened!");
        return true;
    }
    
    return initialize_camera();
}

bool DahengCamera::close_camera()
{
    if (!connected_) {
        return true;
    }
    
    GX_STATUS status = GXCloseDevice(device_handle_);
    if (status != GX_STATUS_SUCCESS) {
        tools::logger()->warn("Failed to close camera: {:#x}", status);
        return false;
    }
    
    device_handle_ = nullptr;
    connected_ = false;
    tools::logger()->info("Camera closed.");
    return true;
}

bool DahengCamera::start_acquisition()
{
    if (!connected_) {
        tools::logger()->error("Camera not connected!");
        return false;
    }
    
    if (capturing_) {
        tools::logger()->warn("Camera already capturing!");
        return true;
    }
    
    // 开始采集
    GX_STATUS status = GXStreamOn(device_handle_);
    DAHENG_CHECK(status);
    
    // 启动采集线程
    capture_quit_ = false;
    capture_thread_ = std::thread(&DahengCamera::capture_thread_func, this);
    
    capturing_ = true;
    tools::logger()->info("Camera acquisition started.");
    return true;
}

bool DahengCamera::stop_acquisition()
{
    if (!capturing_) {
        return true;
    }
    
    // 停止采集线程
    capture_quit_ = true;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
        tools::logger()->debug("Capture thread joined.");
    }
    
    // 停止采集
    if (connected_) {
        GX_STATUS status = GXStreamOff(device_handle_);
        if (status != GX_STATUS_SUCCESS) {
            tools::logger()->warn("Failed to stop acquisition: {:#x}", status);
            return false;
        }
    }
    
    capturing_ = false;
    tools::logger()->info("Camera acquisition stopped.");
    return true;
}

bool DahengCamera::configure_camera()
{
    if (!connected_) {
        return false;
    }
    
    tools::logger()->info("Configuring camera parameters...");
    
    // 设置采集模式为连续采集
    set_enum_value("AcquisitionMode", "Continuous");
    
    // 设置曝光时间
    set_float_value("ExposureTime", exposure_us_);
    
    // 设置增益
    set_float_value("Gain", gain_);
    
    // 设置帧率
    set_float_value("AcquisitionFrameRate", frame_rate_);
    
    // 设置触发模式
    if (trigger_mode_) {
        set_enum_value("TriggerMode", "On");
        set_enum_value("TriggerSource", "Line0");  // 外部触发源
    } else {
        set_enum_value("TriggerMode", "Off");
    }
    
    // 设置白平衡
    if (auto_white_balance_) {
        set_enum_value("BalanceWhiteAuto", "Continuous");
    } else {
        set_enum_value("BalanceWhiteAuto", "Off");
        set_float_value("BalanceRatio", white_balance_r_);
        set_float_value("BalanceRatio", white_balance_g_);
        set_float_value("BalanceRatio", white_balance_b_);
    }
    
    // 获取图像尺寸和格式
    GX_STATUS status;
    int64_t width = 0, height = 0;
    status = GXGetInt(device_handle_, GX_INT_WIDTH, &width);
    DAHENG_CHECK(status);
    status = GXGetInt(device_handle_, GX_INT_HEIGHT, &height);
    DAHENG_CHECK(status);
    
    image_width_ = static_cast<size_t>(width);
    image_height_ = static_cast<size_t>(height);
    
    // 获取像素格式
    int64_t pixel_format = 0;
    status = GXGetEnum(device_handle_, GX_ENUM_PIXEL_FORMAT, &pixel_format);
    DAHENG_CHECK(status);
    pixel_format_ = static_cast<GX_PIXEL_FORMAT_ENTRY>(pixel_format);
    
    tools::logger()->info("Camera configured: {}x{}, pixel format: {}", 
                         width, height, pixel_format);
    
    return true;
}

void DahengCamera::reset_camera()
{
    tools::logger()->warn("Resetting camera connection...");
    
    if (capturing_) {
        stop_acquisition();
    }
    
    if (connected_) {
        close_camera();
    }
    
    // 短暂等待
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // 重新打开相机
    if (open_camera()) {
        start_acquisition();
    }
}

void DahengCamera::capture_thread_func()
{
    tools::logger()->info("Capture thread started.");
    
    // 预分配缓冲区
    PGX_FRAME_BUFFER frame_buffer = nullptr;
    cv::Mat image;
    
    while (!capture_quit_) {
        // 获取图像帧（超时100ms）
        GX_STATUS status = GXGetImage(device_handle_, &frame_buffer, 100);
        
        if (status != GX_STATUS_SUCCESS) {
            if (status == GX_STATUS_TIMEOUT) {
                continue;  // 超时正常，继续尝试
            }
            
            tools::logger()->warn("Failed to get image: {:#x}", status);
            
            // 严重错误，退出采集线程
            capturing_ = false;
            break;
        }
        
        if (frame_buffer == nullptr) {
            continue;
        }
        
        // 转换图像格式
        image = convert_raw_to_mat(frame_buffer->pImgBuf, 
                                  frame_buffer->nWidth, 
                                  frame_buffer->nHeight, 
                                  static_cast<GX_PIXEL_FORMAT_ENTRY>(frame_buffer->nPixelFormat));
        
        if (!image.empty()) {
            // 记录时间戳
            auto timestamp = std::chrono::steady_clock::now();
            
            // 推送数据到队列
            CameraData data{image.clone(), timestamp};
            queue_.push(data);
        }
        
        // 释放帧缓冲区
        status = GXReleaseImageBuffer(device_handle_, frame_buffer);
        if (status != GX_STATUS_SUCCESS) {
            tools::logger()->warn("Failed to release image buffer: {:#x}", status);
        }
        
        // 控制帧率
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    
    tools::logger()->info("Capture thread stopped.");
}

void DahengCamera::daemon_thread_func()
{

}

cv::Mat DahengCamera::convert_raw_to_mat(void* raw_data, size_t width, size_t height, 
                                         GX_PIXEL_FORMAT_ENTRY pixel_format)
{
    if (raw_data == nullptr || width == 0 || height == 0) {
        return cv::Mat();
    }
    
    cv::Mat image;
    
    switch (pixel_format) {
        case GX_PIXEL_FORMAT_MONO8:
            image = cv::Mat(height, width, CV_8UC1, raw_data);
            break;
            
        case GX_PIXEL_FORMAT_MONO10:
        case GX_PIXEL_FORMAT_MONO12:
            // 需要转换为8位
            {
                cv::Mat raw_mat(height, width, CV_16UC1, raw_data);
                raw_mat.convertTo(image, CV_8UC1, 1.0/16.0);  // 12位转8位
            }
            break;
            
        case GX_PIXEL_FORMAT_BAYER_GR8:
        case GX_PIXEL_FORMAT_BAYER_RG8:
        case GX_PIXEL_FORMAT_BAYER_GB8:
        case GX_PIXEL_FORMAT_BAYER_BG8:
            {
                cv::Mat bayer_image(height, width, CV_8UC1, raw_data);
                cv::cvtColor(bayer_image, image, cv::COLOR_BayerBG2RGB);
            }
            break;
            
        case GX_PIXEL_FORMAT_RGB8_PACKED:
            image = cv::Mat(height, width, CV_8UC3, raw_data);
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            break;
            
        case GX_PIXEL_FORMAT_BGR8_PACKED:
            image = cv::Mat(height, width, CV_8UC3, raw_data);
            break;
            
        default:
            tools::logger()->warn("Unsupported pixel format: {}", pixel_format);
            return cv::Mat();
    }
    
    return image;
}

bool DahengCamera::set_float_value(const char* feature_name, double value)
{
    if (!connected_) return false;
    
    GX_STATUS status = GXSetFloat(device_handle_, feature_name, value);
    if (status != GX_STATUS_SUCCESS) {
        tools::logger()->warn("Failed to set {} = {}: {:#x}", feature_name, value, status);
        return false;
    }
    
    tools::logger()->debug("Set {} = {}", feature_name, value);
    return true;
}

bool DahengCamera::set_int_value(const char* feature_name, int64_t value)
{
    if (!connected_) return false;
    
    GX_STATUS status = GXSetInt(device_handle_, feature_name, value);
    if (status != GX_STATUS_SUCCESS) {
        tools::logger()->warn("Failed to set {} = {}: {:#x}", feature_name, value, status);
        return false;
    }
    
    tools::logger()->debug("Set {} = {}", feature_name, value);
    return true;
}

bool DahengCamera::set_enum_value(const char* feature_name, const char* value)
{
    if (!connected_) return false;
    
    GX_STATUS status = GXSetEnumString(device_handle_, feature_name, value);
    if (status != GX_STATUS_SUCCESS) {
        tools::logger()->warn("Failed to set {} = {}: {:#x}", feature_name, value, status);
        return false;
    }
    
    tools::logger()->debug("Set {} = {}", feature_name, value);
    return true;
}

void DahengCamera::set_exposure(double exposure_ms)
{
    exposure_us_ = exposure_ms * 1000.0;
    if (connected_) {
        set_float_value("ExposureTime", exposure_us_);
    }
}

void DahengCamera::set_gain(double gain)
{
    gain_ = gain;
    if (connected_) {
        set_float_value("Gain", gain_);
    }
}

void DahengCamera::set_frame_rate(double frame_rate)
{
    frame_rate_ = frame_rate;
    if (connected_) {
        set_float_value("AcquisitionFrameRate", frame_rate_);
    }
}

void DahengCamera::set_trigger_mode(bool trigger_mode)
{
    trigger_mode_ = trigger_mode;
    if (connected_) {
        if (trigger_mode_) {
            set_enum_value("TriggerMode", "On");
        } else {
            set_enum_value("TriggerMode", "Off");
        }
    }
}

void DahengCamera::set_white_balance(double r_gain, double g_gain, double b_gain)
{
    white_balance_r_ = r_gain;
    white_balance_g_ = g_gain;
    white_balance_b_ = b_gain;
    auto_white_balance_ = false;
    
    if (connected_) {
        set_enum_value("BalanceWhiteAuto", "Off");
        set_float_value("BalanceRatio", r_gain);
        set_float_value("BalanceRatio", g_gain);
        set_float_value("BalanceRatio", b_gain);
    }
}

}  // namespace io