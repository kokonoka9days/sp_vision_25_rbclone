#include "./trt_yolo_0708.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#include "preprocess_kernel.h"   
#include <yaml-cpp/yaml.h>
#include <fmt/format.h>
#include "tools/logger.hpp"
#include "tools/img_tools.hpp"

namespace auto_aim
{

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

void TensorrtInferEngine::softmax(const float* input, float* output, int len) {
    float max_val = *std::max_element(input, input + len);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; ++i) output[i] /= sum;
}

TensorrtInferEngine::TensorrtInferEngine(const string& engine_path, const string& device) {
    loadEngine(engine_path);
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);

    // 获取输入输出张量名称和形状
    input_name = engine->getIOTensorName(0);
    output_name = engine->getIOTensorName(1);
    nvinfer1::Dims input_dims = engine->getTensorShape(input_name.c_str());
    nvinfer1::Dims output_dims = engine->getTensorShape(output_name.c_str());

    cout << "Input name: " << input_name << ", shape: ";
    for (int d = 0; d < input_dims.nbDims; ++d) cout << input_dims.d[d] << " ";
    cout << endl;
    cout << "Output name: " << output_name << ", shape: ";
    for (int d = 0; d < output_dims.nbDims; ++d) cout << output_dims.d[d] << " ";
    cout << endl;

    input_size = 1 * input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * sizeof(float);
    output_size = 1 * output_dims.d[1] * output_dims.d[2] * sizeof(float);
    cout << "Input buffer size: " << input_size << ", Output buffer size: " << output_size << endl;

    cudaMalloc(&input_device, input_size);
    cudaMalloc(&output_device, output_size);

    // 分配页锁定主机内存，用于快速异步拷贝输出
    cudaError_t err = cudaHostAlloc(&d_output_host, output_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Pinned memory allocation failed");
    }

    // 创建 CUDA 事件
    cudaEventCreate(&preprocess_done);
    cudaEventCreate(&inference_done);
    cudaEventCreate(&copy_done);
}

void TensorrtInferEngine::loadEngine(const string& engine_path) {
    ifstream file(engine_path, ios::binary);
    if (!file) throw runtime_error("Cannot open engine file: " + engine_path);
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(data.data(), size);
    delete runtime;
    if (!engine) throw runtime_error("Failed to deserialize engine");
}

TensorrtInferEngine::~TensorrtInferEngine() {
    cudaStreamDestroy(stream);
    cudaFree(input_device);
    cudaFree(output_device);
    
    // 释放复用的输入图像内存
    if (d_img) {
        cudaFree(d_img);
    }

    if (d_output_host) {
        cudaFreeHost(d_output_host);   // 释放页锁定内存
    }
    delete context;
    delete engine;

    cudaEventDestroy(preprocess_done);
    cudaEventDestroy(inference_done);
    cudaEventDestroy(copy_done);
}


void TensorrtInferEngine::infer(Mat img, int detect_color) {
    auto t0 = std::chrono::high_resolution_clock::now();
    objects.clear();
    tmp_objects.clear();

    // 计算当前图像需要的显存大小
    size_t needed_size = img.total() * img.elemSize();
    
    // 如果之前分配的显存不够大，才重新分配（避免每帧 cudaMalloc 带来的严重性能开销）
    if (d_img_size < needed_size) {
        if (d_img) cudaFree(d_img);
        cudaMalloc(&d_img, needed_size);
        d_img_size = needed_size;
    }

    cudaMemcpyAsync(d_img, img.data, needed_size, cudaMemcpyHostToDevice, stream);

    // 启动 CUDA 核函数，直接输出到 input_device
    launch_preprocess(d_img, img.cols, img.rows,
                      (float*)input_device, IMAGE_WIDTH, IMAGE_HEIGHT,
                      stream); 
    
    cudaEventRecord(preprocess_done, stream);   // 记录预处理完成事件
    cudaEventSynchronize(preprocess_done);
    
    // 注意：删除了此处的 cudaFree(d_img); 将其留到下一次或析构时处理

    // 推理
    auto t1 = std::chrono::high_resolution_clock::now();
    context->setTensorAddress(input_name.c_str(), input_device);
    context->setTensorAddress(output_name.c_str(), output_device);
    context->enqueueV3(stream);

    cudaEventRecord(inference_done, stream);    // 记录推理完成事件

    // 直接拷贝到页锁定内存
    cudaMemcpyAsync(d_output_host, output_device, output_size, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(copy_done, stream);         // 记录拷贝完成事件
    cudaStreamSynchronize(stream);

    auto t2 = std::chrono::high_resolution_clock::now();
    
    // 注意：此处解析逻辑强依赖于导出的 ONNX 模型末端是否有 transpose。
    // 如果模型输出shape真的是 [1, 25200, 22] 这样是没问题的。
    const int rows = 25200;
    const int cols = 22;
    const float conf_threshold = 0.65f;
    const float nms_threshold = 0.45f;

    vector<cv::Rect> boxes;
    vector<float> confidences;
    boxes.reserve(rows);
    confidences.reserve(rows);

    float color_probs[4], class_probs[9];
    for (int i = 0; i < rows; ++i) {
        const float* ptr = d_output_host + i * cols;

        float obj_conf = sigmoid(ptr[8]);
        if (obj_conf < conf_threshold) continue;

        softmax(ptr + 9, color_probs, 4);
        int color_id = 0;
        float max_color = color_probs[0];
        for (int k = 1; k < 4; ++k) {
            if (color_probs[k] > max_color) {
                max_color = color_probs[k];
                color_id = k;
            }
        }

        softmax(ptr + 13, class_probs, 9);
        int class_id = 0;
        float max_class = class_probs[0];
        for (int k = 1; k < 9; ++k) {
            if (class_probs[k] > max_class) {
                max_class = class_probs[k];
                class_id = k;
            }
        }

        if (color_id == 2 || color_id == 3) continue;
        if (detect_color == 0 && color_id == 1) continue;
        if (detect_color == 1 && color_id == 0) continue;

        float final_conf = obj_conf * max_class;
        if (final_conf < conf_threshold) continue;

        Object obj;
        obj.prob = final_conf;
        obj.color = color_id;
        obj.label = class_id;

        for (int j = 0; j < 8; ++j) {
            float val = ptr[j];
            obj.landmarks[j] = std::max(0.0f, std::min(640.0f, val));
        }

        // 边界框
        float min_x = obj.landmarks[0], max_x = obj.landmarks[0];
        float min_y = obj.landmarks[1], max_y = obj.landmarks[1];
        for (int j = 2; j < 8; j += 2) {
            min_x = std::min(min_x, obj.landmarks[j]);
            max_x = std::max(max_x, obj.landmarks[j]);
            min_y = std::min(min_y, obj.landmarks[j+1]);
            max_y = std::max(max_y, obj.landmarks[j+1]);
        }
        obj.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);

        cv::Point2f p0(obj.landmarks[0], obj.landmarks[1]);
        cv::Point2f p2(obj.landmarks[2], obj.landmarks[3]);
        cv::Point2f p6(obj.landmarks[6], obj.landmarks[7]);
        obj.length = cv::norm(p0 - p6);
        obj.width = cv::norm(p0 - p2);
        obj.ratio = obj.length / (obj.width + 1e-6f);

        objects.push_back(obj);
        boxes.push_back(obj.rect);
        confidences.push_back(final_conf);
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (int idx : indices) {
        if (idx < (int)objects.size()) tmp_objects.push_back(objects[idx]);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
}

TensorRTYolo::TensorRTYolo(const std::string& config_path, bool debug)
    : debug_(debug)
{
    auto yaml = YAML::LoadFile(config_path);

    std::string engine_path = yaml["tensorrt_engine_path"].as<std::string>();
    std::string device = yaml["device"].as<std::string>("cuda:0");
    min_confidence_ = yaml["min_confidence"].as<double>(0.5);
    detect_color_ = yaml["detect_color"].as<int>(-1);

    // ROI 配置
    use_roi_ = yaml["use_roi"].as<bool>(false);
    if (use_roi_) {
        int x = yaml["roi"]["x"].as<int>(0);
        int y = yaml["roi"]["y"].as<int>(0);
        int w = yaml["roi"]["width"].as<int>(-1);
        int h = yaml["roi"]["height"].as<int>(-1);
        roi_ = cv::Rect(x, y, w, h);
        offset_ = cv::Point2f(x, y);
    }

    engine_ = std::make_unique<TensorrtInferEngine>(engine_path, device);
}

std::list<Armor> TensorRTYolo::detect(const cv::Mat& raw_img, int frame_count)
{
    if (raw_img.empty()) {
        tools::logger()->warn("Empty image in TensorRTYolo::detect");
        return {};
    }

    cv::Mat bgr_img;
    if (use_roi_) {
        // 处理 -1 表示不裁剪
        cv::Rect valid_roi = roi_;
        if (valid_roi.width == -1) valid_roi.width = raw_img.cols - valid_roi.x;
        if (valid_roi.height == -1) valid_roi.height = raw_img.rows - valid_roi.y;
        // 边界检查
        valid_roi.x = std::max(0, valid_roi.x);
        valid_roi.y = std::max(0, valid_roi.y);
        valid_roi.width = std::min(valid_roi.width, raw_img.cols - valid_roi.x);
        valid_roi.height = std::min(valid_roi.height, raw_img.rows - valid_roi.y);
        bgr_img = raw_img(valid_roi);
    } else {
        bgr_img = raw_img;
    }

    // 调用 TensorRT 推理
    engine_->infer(bgr_img, detect_color_);
    // 转换为 Armor 列表
    return convertToArmors(bgr_img, frame_count);
}

std::list<Armor> TensorRTYolo::convertToArmors(const cv::Mat& bgr_img, int frame_count)
{
    std::list<Armor> armors;
    const auto& objects = engine_->tmp_objects;

    // 计算缩放因子
    float sx = static_cast<float>(bgr_img.cols) / engine_->IMAGE_WIDTH;
    float sy = static_cast<float>(bgr_img.rows) / engine_->IMAGE_HEIGHT;

    for (const auto& obj : objects) {
        if (detect_color_ == 0 && obj.color != 0) continue;
        if (detect_color_ == 1 && obj.color != 1) continue;

        std::vector<cv::Point2f> pts;
        for (int i = 0; i < 8; i += 2) {
            float x = obj.landmarks[i] * sx;
            float y = obj.landmarks[i+1] * sy;
            pts.emplace_back(x, y);
        }
        sortKeypoints(pts);

        // 缩放边界框
        cv::Rect scaled_box(obj.rect.x * sx, obj.rect.y * sy, 
                            obj.rect.width * sx, obj.rect.height * sy);

        // 颜色映射修复：模型输出(0红, 1蓝) 映射到 Armor 构造函数期望的 color_id (0蓝, 1红)
        int mapped_color_id = (obj.color == 0) ? 1 : (obj.color == 1) ? 0 : 2;

        
        Armor armor = use_roi_ 
            ? Armor(mapped_color_id, obj.label, obj.prob, scaled_box, pts, offset_)
            : Armor(mapped_color_id, obj.label, obj.prob, scaled_box, pts);

        // 过滤和合法性检查
        if (armor.confidence < min_confidence_) continue;
        if (!checkName(armor)) continue;
        if (!checkType(armor)) continue;

        // 计算归一化
        armor.center_norm = cv::Point2f(armor.center.x / (bgr_img.cols + (use_roi_ ? offset_.x : 0)),
                                        armor.center.y / (bgr_img.rows + (use_roi_ ? offset_.y : 0)));

        armors.push_back(armor);
    }

    if (debug_) drawDetections(bgr_img, armors, frame_count);
    return armors;
}

void TensorRTYolo::sortKeypoints(std::vector<cv::Point2f>& pts)
{
    if (pts.size() != 4) return;
    // 按 y 坐标排序，分出上下两对
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    cv::Point2f top_left = pts[0].x < pts[1].x ? pts[0] : pts[1];
    cv::Point2f top_right = pts[0].x < pts[1].x ? pts[1] : pts[0];
    cv::Point2f bottom_left = pts[2].x < pts[3].x ? pts[2] : pts[3];
    cv::Point2f bottom_right = pts[2].x < pts[3].x ? pts[3] : pts[2];
    pts = {top_left, top_right, bottom_right, bottom_left};
}

bool TensorRTYolo::checkName(const Armor& armor) const
{
    return armor.name != ArmorName::not_armor;
}

bool TensorRTYolo::checkType(const Armor& armor) const
{
    if (armor.type == ArmorType::small) {
        return armor.name != ArmorName::one && armor.name != ArmorName::base;
    } else {
        return armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
               armor.name != ArmorName::outpost;
    }
}

ArmorType TensorRTYolo::getType(const Armor& armor) const
{
    if (armor.name == ArmorName::one || armor.name == ArmorName::base)
        return ArmorType::big;
    if (armor.name == ArmorName::two || armor.name == ArmorName::sentry ||
        armor.name == ArmorName::outpost)
        return ArmorType::small;
    return ArmorType::small;
}

void TensorRTYolo::drawDetections(const cv::Mat& img, const std::list<Armor>& armors,
                                  int frame_count) const
{
    auto detection = img.clone();
    // tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255,255,255});
    for (const auto& armor : armors) {
        std::string info = fmt::format("{:.2f} {} {}", armor.confidence,
                                       ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
        tools::draw_points(detection, armor.points, {0,255,255});
        tools::draw_text(detection, info, armor.center, {0,255,255});
    }
    if (use_roi_) {
        cv::rectangle(detection, roi_, {0,255,0}, 2);
    }
    cv::resize(detection, detection, {}, 0.5, 0.5);
    cv::imshow("TensorRT Detection", detection);
}

std::list<Armor> TensorRTYolo::postprocess(double scale, cv::Mat& output,
                                           const cv::Mat& bgr_img, int frame_count)
{
    return {};  // TensorRT 这里不调用
}

} // namespace auto_aim