#include "drone_yolo.hpp"
#include <fstream>
#include <algorithm>
#include <stdexcept>

// 【新增】引入 OpenVINO 预处理模块
#include <openvino/core/preprocess/pre_post_process.hpp> 

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace auto_drone
{

// ==========================================
// 构造函数: 加载模型，配置硬件级预处理并编译
// ==========================================
YOLO::YOLO(const std::string& config_path, bool /*debug*/) 
{
    auto yaml = tools::load(config_path);
    std::string model_path = tools::read<std::string>(yaml, "model_path"); 
    
    this->input_w_ = 640;
    this->input_h_ = 640;
    this->score_threshold_ = 0.70f;
    this->nms_threshold_ = 0.6f;
    
    this->num_classes_ = 1; 
    this->num_kpts_ = 8;    
    this->num_boxes_ = 8400; 
    
    try {
        // 1. 读取 ONNX / OpenVINO IR 模型
        std::shared_ptr<ov::Model> model = core_.read_model(model_path);
        
        // ========================================================
        // 【新增优化】OpenVINO PrePostProcessor (PPP) 硬件预处理加速
        // 将耗时的 BGR2RGB、归一化、HWC2CHW 从 CPU for循环 中剥离
        // ========================================================
        ov::preprocess::PrePostProcessor ppp(model);
        
        // A. 声明输入数据格式 (来自 OpenCV 的 Mat: 类型为 uint8, 布局 NHWC, BGR)
        ov::preprocess::InputInfo& input_info = ppp.input(0);
        input_info.tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
            
        // B. 声明预处理步骤 (转换为模型需要的 F32，色彩转 RGB，除以 255.0 归一化)
        input_info.preprocess()
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .convert_element_type(ov::element::f32)
            .scale(255.0f); // 缩放因子：等效于除以 255
            
        // C. 声明模型内部实际需要的布局 (YOLO 需要 NCHW)
        input_info.model().set_layout("NCHW");
        
        // 构建带有预处理管线的新模型
        model = ppp.build();
        
        // ========================================================
        // 【新增优化】配置 Performance Hints 加速推理延迟
        // ========================================================
        ov::AnyMap config;
        config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)); // 追求单次推理的极低延迟
        
        // 2. 编译模型：AUTO 会优先选择 GPU (核显)，如果没有再使用 CPU
        // 使用 "AUTO:GPU,CPU" 明确告知设备优先级
        this->compiled_model_ = core_.compile_model(model, "AUTO:GPU,CPU", config);
        
        // 3. 创建推理请求
        this->infer_request_ = compiled_model_.create_infer_request();
        
        tools::logger()->info("auto_drone::YOLO OpenVINO Engine Loaded ONNX Successfully with PPP optimization.");
    } 
    catch (const std::exception& e) {
        tools::logger()->error("[YOLO] OpenVINO init error: {}", e.what());
        throw std::runtime_error("Failed to initialize OpenVINO model.");
    }
}

// ==========================================
// 解码器 (后处理) - 保持不变
// ==========================================
std::vector<Drone> YOLO::postprocessing(float* output) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> valid_raw_indices;

    int stride = this->num_boxes_; 
    int kpt_offset = 4 + this->num_classes_; 

    for (int i = 0; i < this->num_boxes_; i++) {
        float max_conf = 0.0f;
        int max_id = -1;
        for (int c = 0; c < this->num_classes_; c++) {
            float conf = output[(4 + c) * stride + i];
            if (conf > max_conf) {
                max_conf = conf;
                max_id = c;
            }
        }

        if (max_conf < this->score_threshold_) continue;

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

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->score_threshold_, this->nms_threshold_, indices);

    std::vector<Drone> results;
    for (int idx : indices) {
        int raw_i = valid_raw_indices[idx];
        
        std::vector<cv::Point2f> kpts;
        kpts.reserve(this->num_kpts_);

        float inv_scale = 1.0f / this->scale_;
        for (int k = 0; k < this->num_kpts_; ++k) {
            float kx = output[(kpt_offset + k * 3 + 0) * stride + raw_i];
            float ky = output[(kpt_offset + k * 3 + 1) * stride + raw_i];

            float raw_kx = (kx - this->pad_w_) * inv_scale;
            float raw_ky = (ky - this->pad_h_) * inv_scale;

            kpts.push_back(cv::Point2f(raw_kx, raw_ky));
        }

        results.emplace_back(
            classIds[idx], 
            confidences[idx], 
            boxes[idx], 
            kpts
        );
    }

    if (!results.empty()) {
        auto best_it = std::max_element(results.begin(), results.end(),
            [](const Drone& a, const Drone& b) {
                return a.confidence < b.confidence;
            });
        return { *best_it };
    }

    return results;
}

// ==========================================
// 核心推理流程
// ==========================================
std::vector<Drone> YOLO::detect(const cv::Mat &frame) {
    if (frame.empty()) return {};

    // 1. CPU 端完成自适应缩放填充 (Letterbox)
    this->scale_ = std::min((float)this->input_w_ / frame.cols, (float)this->input_h_ / frame.rows);
    this->pad_w_ = (this->input_w_ - frame.cols * this->scale_) / 2;
    this->pad_h_ = (this->input_h_ - frame.rows * this->scale_) / 2;

    cv::Mat resized_img, pad_img;
    cv::resize(frame, resized_img, cv::Size(), this->scale_, this->scale_);
    cv::copyMakeBorder(resized_img, pad_img, 
                       this->pad_h_, this->input_h_ - resized_img.rows - this->pad_h_,
                       this->pad_w_, this->input_w_ - resized_img.cols - this->pad_w_, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // ========================================================
    // 【修改优化】使用 Zero-Copy 机制直接封送 OpenCV 内存
    // ========================================================
    // 直接复用 pad_img 的连续内存。因为我们在 init 中配置了 PPP 预处理，
    // OpenVINO 知道这块内存是 NHWC / BGR / uint8 格式，它会自动帮我们转换为模型需要的格式。
    ov::Tensor input_tensor(ov::element::u8, 
                            {1, (size_t)this->input_h_, (size_t)this->input_w_, 3}, 
                            pad_img.data);
                            
    this->infer_request_.set_input_tensor(input_tensor);

    // 3. 执行同步前向推理
    this->infer_request_.infer();

    // 4. 获取输出 Tensor 并交由后处理解码 
    ov::Tensor output_tensor = this->infer_request_.get_output_tensor();
    float* output_data = output_tensor.data<float>();

    return this->postprocessing(output_data);
}
}