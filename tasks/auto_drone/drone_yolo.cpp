#include "drone_yolo.hpp"
#include <fstream>
#include <algorithm>

#include "cuda_preprocess.cuh" 
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{

// TensorRT Logger 实例
class YOLOLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if(severity <= Severity::kWARNING)
            tools::logger()->warn("[TRT YOLO] {}", msg);
    }     
} gYOLOLogger;

// ==========================================
// 构造函数
// ==========================================
YOLO::YOLO(const std::string& config_path, bool debug) 
{
    // 1. 读取配置文件 (若 YAML 中没有，这里给出默认值)
    auto yaml = tools::load(config_path);
    std::string model_path = tools::read<std::string>(yaml, "model_path"); // 请确保您的 yaml 文件里有这个键
    
    this->input_w_ = 640;
    this->input_h_ = 640;
    this->score_threshold_ = 0.5f;
    this->nms_threshold_ = 0.45f;
    this->num_classes_ = 2; // 0: blue, 1: red
    this->num_kpts_ = 8;    // 无人机 8 个关键点
    this->use_cuda_preproc_ = true; 
    
    // 2. 加载 TensorRT Engine 文件
    std::ifstream engineFile(model_path, std::ios::binary);
    if(!engineFile.good()) throw std::runtime_error("Error cannot open engine file: " + model_path);
    
    engineFile.seekg(0, engineFile.end);
    size_t fsize = engineFile.tellg();  
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();

    // 3. 初始化 TensorRT 核心对象
    this->runtime_ = nvinfer1::createInferRuntime(gYOLOLogger);
    this->engine_  = this->runtime_->deserializeCudaEngine(engineData.data(), fsize);
    this->context_ = this->engine_->createExecutionContext();
    cudaStreamCreate(&this->stream_);

    // 4. 获取输入/输出维度信息
    nvinfer1::Dims input_dims = this->engine_->getBindingDimensions(0);
    nvinfer1::Dims output_dims = this->engine_->getBindingDimensions(1);

    // 计算输入输出的内存大小
    int input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) input_size_ *= input_dims.d[i];
    
    this->output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) this->output_size_ *= output_dims.d[i];
    this->num_boxes_ = output_dims.d[2]; // 通常是 8400

    // 5. 分配 Pinned Memory (主机端锁页内存) 与 Device Memory (显存)
    cudaMallocHost((void**)&this->pinned_in_host_, input_size_ * sizeof(float));
    cudaMallocHost((void**)&this->pinned_out_host_, this->output_size_ * sizeof(float));

    cudaMalloc(&this->buffer_idx_0_, input_size_ * sizeof(float));
    cudaMalloc(&this->buffer_idx_1_, this->output_size_ * sizeof(float));

    // 预分配用于图像拷贝的显存 (假设最大分辨率为 1440x1080)
    this->max_src_size_ = 1440 * 1080 * 3;
    cudaMalloc((void**)&this->d_src_img_, this->max_src_size_ * sizeof(uint8_t));

    tools::logger()->info("auto_drone::YOLO TensorRT Engine Loaded Successfully.");
}

// ==========================================
// 析构函数
// ==========================================
YOLO::~YOLO() {
    cudaStreamSynchronize(this->stream_);
    cudaStreamDestroy(this->stream_);

    if(this->buffer_idx_0_) cudaFree(this->buffer_idx_0_);
    if(this->buffer_idx_1_) cudaFree(this->buffer_idx_1_);
    if(this->d_src_img_)    cudaFree(this->d_src_img_);

    if(this->pinned_in_host_)  cudaFreeHost(this->pinned_in_host_);
    if(this->pinned_out_host_) cudaFreeHost(this->pinned_out_host_);

    if(this->context_) this->context_->destroy();
    if(this->engine_)  this->engine_->destroy();
    if(this->runtime_) this->runtime_->destroy();
}

// ==========================================
// CUDA 预处理
// ==========================================
void YOLO::preprocess_cuda(const cv::Mat &frame) {
    // 1. 计算缩放与 Padding 比例
    this->scale_ = std::min((float)this->input_w_ / frame.cols, (float)this->input_h_ / frame.rows);
    this->pad_w_ = (this->input_w_ - frame.cols * this->scale_) / 2;
    this->pad_h_ = (this->input_h_ - frame.rows * this->scale_) / 2;

    int src_size = frame.cols * frame.rows * 3;
    if (src_size > this->max_src_size_) {
        cudaFree(this->d_src_img_);
        this->max_src_size_ = src_size * 1.5;
        cudaMalloc((void**)&this->d_src_img_, this->max_src_size_ * sizeof(uint8_t));
    }

    // 2. 将原图拷贝进显存
    cudaMemcpyAsync(this->d_src_img_, frame.data, src_size, cudaMemcpyHostToDevice, this->stream_);

    // 3. 调用 CUDA 预处理核函数
    launch_preprocess_kernel(
        this->d_src_img_, frame.step, frame.cols, frame.rows,
        (float*)this->buffer_idx_0_, this->input_w_, this->input_h_,
        this->scale_, this->pad_w_, this->pad_h_, 
        this->stream_
    );
}

// ==========================================
// 解码器 (后处理)
// ==========================================
std::vector<Drone> YOLO::postprocessing() {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> valid_raw_indices;

    float* output = this->pinned_out_host_;
    int stride = this->num_boxes_; // 行跨度 (即输出宽度 8400)
    int kpt_offset = 4 + this->num_classes_; 

    // YOLOv8/11 Pose 输出维度: [1, 4(bbox) + nc(cls) + nk*3(kpts), 8400]
    // 在内存中由于是 CHW 平铺，可以通过 output[channel * stride + anchor_idx] 提取
    for (int i = 0; i < this->num_boxes_; i++) {
        // 1. 获取类别置信度
        float max_conf = 0.0f;
        int max_id = -1;
        for (int c = 0; c < this->num_classes_; c++) {
            float conf = output[(4 + c) * stride + i];
            if (conf > max_conf) {
                max_conf = conf;
                max_id = c;
            }
        }

        // 置信度过滤
        if (max_conf < this->score_threshold_) continue;

        // 2. 获取并还原边界框坐标
        float cx = output[0 * stride + i];
        float cy = output[1 * stride + i];
        float w  = output[2 * stride + i];
        float h  = output[3 * stride + i];

        float inv_scale = 1.0f / this->scale_;
        int raw_w = static_cast<int>(w * inv_scale);
        int raw_h = static_cast<int>(h * inv_scale);
        int raw_x = static_cast<int>((cx - this->pad_w_) * inv_scale - raw_w / 2.0f);
        int raw_y = static_cast<int>((cy - this->pad_h_) * inv_scale - raw_h / 2.0f);

        boxes.push_back(cv::Rect(raw_x, raw_y, raw_w, raw_h));
        confidences.push_back(max_conf);
        classIds.push_back(max_id);
        valid_raw_indices.push_back(i);
    }

    // 3. 执行 NMS (非极大值抑制)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->score_threshold_, this->nms_threshold_, indices);

    // 4. 解析真实目标关键点
    std::vector<Drone> results;
    for (int idx : indices) {
        int raw_i = valid_raw_indices[idx];
        
        std::vector<cv::Point2f> kpts;
        kpts.reserve(this->num_kpts_);

        float inv_scale = 1.0f / this->scale_;
        for (int k = 0; k < this->num_kpts_; ++k) {
            float kx = output[(kpt_offset + k * 3 + 0) * stride + raw_i];
            float ky = output[(kpt_offset + k * 3 + 1) * stride + raw_i];
            // float kv = output[(kpt_offset + k * 3 + 2) * stride + raw_i]; // 可见度

            float raw_kx = (kx - this->pad_w_) * inv_scale;
            float raw_ky = (ky - this->pad_h_) * inv_scale;

            kpts.push_back(cv::Point2f(raw_kx, raw_ky));
        }

        // 组装最终 Drone 对象并填入
        results.emplace_back(
            classIds[idx], 
            confidences[idx], 
            boxes[idx], 
            kpts
        );
    }

    return results;
}

// ==========================================
// 核心推理流程
// ==========================================
std::vector<Drone> YOLO::detect(const cv::Mat &frame) {
    if (frame.empty()) return {};

    // 1. 预处理 (Host -> Device + CUDA Kernel)
    this->preprocess_cuda(frame);

    // 2. TensorRT 异步推理
    void* bindings[] = {this->buffer_idx_0_, this->buffer_idx_1_};
    this->context_->enqueueV2(bindings, this->stream_, nullptr);

    // 3. 推理结果回拷 (Device -> Host Pinned Memory)
    cudaMemcpyAsync(this->pinned_out_host_, this->buffer_idx_1_, 
                    this->output_size_ * sizeof(float), cudaMemcpyDeviceToHost, this->stream_);

    // 4. 流同步，确保推理和回拷完成
    cudaStreamSynchronize(this->stream_);

    // 5. 后处理解码 (CPU 端)
    return this->postprocessing();
}

} // namespace auto_drone