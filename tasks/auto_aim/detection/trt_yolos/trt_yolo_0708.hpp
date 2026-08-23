#ifndef AUTO_AIM__TENSORRT_YOLO_HPP
#define AUTO_AIM__TENSORRT_YOLO_HPP

#include <list>
#include <string>
#include <opencv2/opencv.hpp>

#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <mutex>

#include "../../model/armor.hpp"
#include "../yolo.hpp"
#include "trt_execution_slot.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace cv;
using namespace std;

namespace auto_aim
{
struct Object {
    cv::Rect_<float> rect;
    float landmarks[8];
    int label;
    float prob;
    int color;      // 模型输出: blue:1 , red:0
    double length;
    double width;
    double ratio;
};

struct TRTFrameData : public YOLOFrameData, public TRTExecutionSlot<float> {
    int frame_id = -1;

    std::vector<Object> detected_objects; // 存放该帧的解析结果

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

class TensorrtInferEngine {
public:
    vector<Object> objects;
    vector<Object> tmp_objects;
    const int IMAGE_HEIGHT = 640;
    const int IMAGE_WIDTH = 640;

    /** @brief 加载 TensorRT 引擎并初始化执行槽 @param engine_path 引擎文件路径 @param device CUDA 设备标识 */
    TensorrtInferEngine(const string& engine_path, const string& device = "cuda:0");
    /** @brief 停止工作线程并释放 TensorRT 资源 */
    ~TensorrtInferEngine();
    /** @brief 同步执行一帧推理 @param img 输入图像 @param detect_color 颜色过滤值 */
    void infer(Mat img, int detect_color);
    /** @brief 向异步流水线提交帧并尝试取回结果 @param bgr_img BGR 图像 @param frame_info 帧元数据 @param frame_count 帧编号 @param detect_color 颜色过滤值 @param out_objects 输出候选对象 @param out_frame_data 输出帧数据 @return 取得完成结果时返回 true */
    bool async_enabled_infer(const cv::Mat& bgr_img, const YOLOFrameData& frame_info, int frame_count, int detect_color, std::vector<Object>& out_objects, YOLOFrameData& out_frame_data);

    /** @brief 解码网络输出并执行筛选 @param output_ptr 网络输出缓冲区 @param detect_color 颜色过滤值 @return 候选对象列表 */
    std::vector<Object> decode_outputs(const float* output_ptr, int detect_color);
private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    tools::ThreadSafeQueue<TRTFrameData> task_queue; // 待推理队列
    tools::ThreadSafeQueue<TRTFrameData> free_buffer_queue; // 空闲显存池队列
    tools::ThreadSafeQueue<TRTFrameData> result_queue;      // 结果队列

    size_t input_size = 0;
    size_t output_size = 0;
    string input_name;
    string output_name;

    TRTExecutionSlot<float> sync_slot_;

    std::thread infer_work_thread;
    int in_flight_count = 0; // 记录当前在 GPU 中飞行的帧数（用于流水线控制）
    std::mutex sync_mutex_;
    /** @brief 在指定执行槽上启动推理 @param slot TensorRT 执行槽 */
    void execute(TRTExecutionSlot<float>& slot);
    /** @brief 异步推理工作线程入口 */
    void infer_workerLoop();

    /** @brief 从文件反序列化 TensorRT 引擎 @param engine_path 引擎文件路径 */
    void loadEngine(const string& engine_path);
    /** @brief 计算数值稳定的 sigmoid @param x 输入值 @return sigmoid 结果 */
    double sigmoid(double x) {
        if (x > 0) return 1.0 / (1.0 + exp(-x));
        else return exp(x) / (1.0 + exp(x));
    }
    /** @brief 计算 softmax @param input 输入数组 @param output 输出数组 @param len 元素数 */
    static void softmax(const float* input, float* output, int len);
};

class TensorRTYolo0708 : public YOLOBase
{
public:
    /** @brief 根据配置创建 0708 TensorRT YOLO 检测器 @param config_path YAML 配置路径 @param debug 是否启用调试输出 */
    TensorRTYolo0708(const std::string& config_path, bool debug = true);
    /** @brief 销毁检测器 */
    ~TensorRTYolo0708() = default;

    /** @brief 检测图像中的装甲板 @param img 输入图像 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> detect(const cv::Mat& img, int frame_count = -1) override;

    /** @brief 异步检测带元数据帧 @param frame_data 帧数据 @param frame_count 帧编号 @return 完成的帧结果 */
    YOLOFrameData detect(YOLOFrameData frame_data, int frame_count = -1) override;

    /** @brief 兼容 YOLOBase 的后处理接口 @param scale 缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> postprocess(double scale, cv::Mat& output,
                                 const cv::Mat& bgr_img, int frame_count) override;

private:
    std::unique_ptr<TensorrtInferEngine> engine_;
    bool debug_;
    bool use_roi_;
    cv::Rect roi_;
    cv::Point2f offset_;

    double min_confidence_;
    int detect_color_;          // -1: all, 0: red, 1: blue

    /** @brief 将引擎候选对象转换为装甲板 @param inference_img 推理输入图像 @param raw_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
    std::list<Armor> convertToArmors(
      const cv::Mat& inference_img, const cv::Mat& raw_img, int frame_count);
    /** @brief 统一关键点顺序 @param pts 关键点 */
    void sortKeypoints(std::vector<cv::Point2f>& pts);
    /** @brief 检查类别名称 @param armor 装甲板 @return 合法时返回 true */
    bool checkName(const Armor& armor) const;
    /** @brief 检查尺寸类型 @param armor 装甲板 @return 合法时返回 true */
    bool checkType(const Armor& armor) const;
    /** @brief 根据几何比例判断装甲板尺寸 @param armor 装甲板 @return 尺寸类型 */
    ArmorType getType(const Armor& armor) const;
    /** @brief 绘制检测结果 @param img 图像 @param armors 装甲板列表 @param frame_count 帧编号 */
    void drawDetections(const cv::Mat& img, const std::list<Armor>& armors, int frame_count) const;
};

} // namespace auto_aim

#endif // AUTO_AIM__TENSORRT_YOLO_HPP
