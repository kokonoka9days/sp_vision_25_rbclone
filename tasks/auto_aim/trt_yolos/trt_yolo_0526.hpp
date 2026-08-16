#ifndef AUTO_AIM__TRT_YOLO_0526_HPP
#define AUTO_AIM__TRT_YOLO_0526_HPP

#include "../yolo.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include "trt_execution_slot.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim {

// 检测结果结构
struct Object0526 {
    cv::Rect_<float> rect;
    float landmarks[8];
    int digit_id;      // 数字类别 0~8
    int color_id;      // 颜色类别 (0=红, 1=蓝, 按模型定义)
    float prob;
    double length, width, ratio;
};

// 帧数据（含 GPU 资源）
struct TRTFrameData0526 : public YOLOFrameData, public TRTExecutionSlot<half> {
    int frame_id = -1;

    std::vector<Object0526> detected_objects;

    /** @brief 复制通用帧元数据到 TensorRT 帧 @param f 通用帧数据 */
    void setTRTFrameData(const YOLOFrameData& f) {
        this->frame = f.frame;
        this->gimbal_q = f.gimbal_q;
        this->timestamp = f.timestamp;
        this->armors = f.armors;
        this->detect_color = f.detect_color;
        this->is_empty = false;
    }
};

// 异步推理引擎
class TensorrtInferEngine0526 {
public:
    static constexpr int IMAGE_WIDTH  = 640;
    static constexpr int IMAGE_HEIGHT = 640;

    /** @brief 加载 TensorRT 引擎并初始化执行槽 @param engine_path 引擎文件路径 @param device CUDA 设备标识 */
    TensorrtInferEngine0526(const std::string& engine_path, const std::string& device = "cuda:0");
    /** @brief 停止工作线程并释放 TensorRT 资源 */
    ~TensorrtInferEngine0526();

    /** @brief 同步执行一帧推理 @param bgr_img BGR 图像 @param frame_count 帧编号 @param detect_color 颜色过滤值 @param out_objects 输出候选对象 @param out_frame_data 输出帧数据 */
    void infer(const cv::Mat& bgr_img,
                                    int frame_count,
                                    int detect_color,
                                    std::vector<Object0526>& out_objects,
                                    YOLOFrameData& out_frame_data);
                            
    /** @brief 向异步流水线提交帧并尝试取回结果 @param bgr_img BGR 图像 @param frame_info 帧元数据 @param frame_count 帧编号 @param detect_color 颜色过滤值 @param out_objects 输出候选对象 @param out_frame_data 输出帧数据 @return 取得完成结果时返回 true */
    bool async_enabled_infer(const cv::Mat& bgr_img,
                             const YOLOFrameData& frame_info,
                             int frame_count,
                             int detect_color,
                             std::vector<Object0526>& out_objects,
                             YOLOFrameData& out_frame_data);

private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    tools::ThreadSafeQueue<TRTFrameData0526> task_queue_;
    tools::ThreadSafeQueue<TRTFrameData0526> free_buffer_queue_;
    tools::ThreadSafeQueue<TRTFrameData0526> result_queue_;

    std::thread worker_thread_;
    int in_flight_count_ = 0;

    size_t input_size_ = 0, output_size_ = 0;
    std::string input_name_, output_name_;

    /** @brief 异步推理工作线程入口 */
    void workerLoop();
    /** @brief 在指定执行槽上启动推理 @param slot TensorRT 执行槽 */
    void execute(TRTExecutionSlot<half>& slot);
    /** @brief 解码网络输出并执行筛选 @param output_ptr 网络输出缓冲区 @param detect_color 颜色过滤值 @return 候选对象列表 */
    std::vector<Object0526> decode_outputs(const float* output_ptr, int detect_color);

    /** @brief 计算 sigmoid @param x 输入值 @return sigmoid 结果 */
    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    /** @brief 计算 softmax @param input 输入数组 @param output 输出数组 @param len 元素数 */
    static void softmax(const float* input, float* output, int len);

    float conf_threshold_ = 0.65f;
    float nms_threshold_ = 0.45f;
};

// YOLO 封装类
class TensorRTYolo0526 : public YOLOBase {
public:
    /** @brief 根据配置创建 0526 TensorRT YOLO 检测器 @param config_path YAML 配置路径 @param debug 是否启用调试输出 */
    explicit TensorRTYolo0526(const std::string& config_path, bool debug = true);
    /** @brief 销毁检测器 */
    ~TensorRTYolo0526() = default;

    /** @brief 检测图像中的装甲板 @param img 输入图像 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> detect(const cv::Mat& img, int frame_count = -1) override;
    /** @brief 异步检测带元数据帧 @param frame_data 帧数据 @param frame_count 帧编号 @return 完成的帧结果 */
    YOLOFrameData detect(YOLOFrameData frame_data, int frame_count = -1) override;
    /** @brief 兼容 YOLOBase 的后处理接口 @param scale 缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> postprocess(double scale, cv::Mat& output,
                                 const cv::Mat& bgr_img, int frame_count) override;

private:
    std::unique_ptr<TensorrtInferEngine0526> engine_;
    bool debug_;
    int detect_color_;
    double min_confidence_;

    /** @brief 将 TensorRT 候选对象转换为装甲板 @param objects 候选对象 @param img 原图 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> convertToArmors(const std::vector<Object0526>& objects,
                                     const cv::Mat& img,
                                     int frame_count);
    /** @brief 统一关键点顺序 @param pts 关键点 */
    void sortKeypoints(std::vector<cv::Point2f>& pts);
    /** @brief 检查类别名称 @param armor 装甲板 @return 合法时返回 true */
    bool checkName(const Armor& armor) const;
    /** @brief 检查尺寸类型 @param armor 装甲板 @return 合法时返回 true */
    bool checkType(const Armor& armor) const;
    /** @brief 绘制检测结果 @param img 图像 @param armors 装甲板列表 @param frame_count 帧编号 */
    void drawDetections(const cv::Mat& img, const std::list<Armor>& armors, int frame_count) const;
};

} // namespace auto_aim

#endif
