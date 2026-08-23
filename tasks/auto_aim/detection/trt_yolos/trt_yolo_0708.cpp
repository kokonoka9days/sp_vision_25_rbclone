#include "./trt_yolo_0708.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#include "trt_0708_kernel.h"   
#include <yaml-cpp/yaml.h>
#include <fmt/format.h>
#include "tools/logger.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "../roi.hpp"

#define IS_ASYNC true

namespace auto_aim
{

namespace {
class Logger : public nvinfer1::ILogger {
    /** @brief 输出 TensorRT 警告及更严重日志 @param severity 严重级别 @param msg 日志文本 */
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;
} // namespace
void TensorrtInferEngine::softmax(const float* input, float* output, int len) {
    float max_val = *std::max_element(input, input + len);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; ++i) output[i] /= sum;
}

TensorrtInferEngine::TensorrtInferEngine(const string& engine_path, const string& device)
    : task_queue(3, tools::OverflowPolicy::Block),
      free_buffer_queue(3, tools::OverflowPolicy::Block),
      result_queue(3, tools::OverflowPolicy::Block) {
    loadEngine(engine_path);


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
    
    if(IS_ASYNC){
        // 初始化缓冲池 (三缓冲)
        const int POOL_SIZE = 3;
        for (int i = 0; i < POOL_SIZE; ++i) {
            TRTFrameData task;
            task.initialize(engine.get(), input_size, output_size);
            free_buffer_queue.push(std::move(task));
        }        
        infer_work_thread = std::thread(&TensorrtInferEngine::infer_workerLoop, this);
    }

    // The synchronous facade owns a separate context and buffers.
    sync_slot_.initialize(engine.get(), input_size, output_size);
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

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) throw runtime_error("Failed to create TensorRT runtime");
    engine.reset(runtime->deserializeCudaEngine(data.data(), size));
    if (!engine) throw runtime_error("Failed to deserialize engine");
}
TensorrtInferEngine::~TensorrtInferEngine() {

    if(IS_ASYNC){
        task_queue.close();
        result_queue.close();
        if (infer_work_thread.joinable()) {
            infer_work_thread.join();
        }

        free_buffer_queue.close();
        free_buffer_queue.clear();
        task_queue.clear();
        result_queue.clear();
    }

    sync_slot_.reset();
}


void TensorrtInferEngine::infer(Mat img, int detect_color) {
    std::lock_guard<std::mutex> sync_lock(sync_mutex_);
    auto & slot = sync_slot_;
    auto t0 = std::chrono::steady_clock::now();
    objects.clear();
    tmp_objects.clear();

    // 计算当前图像需要的显存大小
    size_t needed_size = img.total() * img.elemSize();
    
    // 如果之前分配的显存不够大，才重新分配（避免每帧 cudaMalloc 带来的严重性能开销）
    slot.ensure_image_capacity(needed_size);

    check_cuda(
      cudaMemcpyAsync(slot.d_img, img.data, needed_size, cudaMemcpyHostToDevice, slot.stream),
      "cudaMemcpyAsync(image)");

    // 启动 CUDA 核函数，直接输出到 input_device
    launch_preprocess(slot.d_img, img.cols, img.rows,
                      slot.input_device, IMAGE_WIDTH, IMAGE_HEIGHT,
                      slot.stream);
    check_cuda(cudaGetLastError(), "launch_preprocess");
    
    // 注意：删除了此处的 cudaFree(d_img); 将其留到下一次或析构时处理

    // 推理
    auto t1 = std::chrono::steady_clock::now();
    execute(slot);

    auto t2 = std::chrono::steady_clock::now();
    
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
        const float* ptr = slot.output_host + i * cols;

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
        obj.color = !color_id;
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

    auto t3 = std::chrono::steady_clock::now();

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (int idx : indices) {
        if (idx < (int)objects.size()) tmp_objects.push_back(objects[idx]);
    }
    auto t4 = std::chrono::steady_clock::now();

    // tools::logger()->info("耗时   预处理: {:.3f}ms,    推理: {:.3f}ms,     后处理: {:.3f}ms,    nms: {:.3f}ms", 
    //     tools::delta_time(t1, t0)*1000, 
    //     tools::delta_time(t2, t1)*1000,
    //     tools::delta_time(t3, t2)*1000,
    //     tools::delta_time(t4, t3)*1000);
}

bool TensorrtInferEngine::async_enabled_infer(const cv::Mat& bgr_img, const YOLOFrameData& frame_info, int frame_count, int detect_color, std::vector<Object>& out_objects, YOLOFrameData& out_frame_data) {
    
    // 1. 从空闲池获取一个 Buffer
    auto available_task = free_buffer_queue.wait_pop();
    if (!available_task.has_value()) return false;
    TRTFrameData task = std::move(*available_task);
    task.setTRTFrameData(frame_info);
    task.frame_id = frame_count;
    task.detect_color = detect_color;

    // 2. 动态调整设备显存大小并异步拷贝图像
    size_t needed_size = bgr_img.total() * bgr_img.elemSize();
    task.ensure_image_capacity(needed_size);
    
    check_cuda(
      cudaMemcpyAsync(
        task.d_img, bgr_img.data, needed_size, cudaMemcpyHostToDevice, task.stream),
      "cudaMemcpyAsync(image)");

    // 3. 启动 CUDA 预处理核函数（非阻塞，交由 Stream 自动排队）
    launch_preprocess(task.d_img, bgr_img.cols, bgr_img.rows,
                      (float*)task.input_device, IMAGE_WIDTH, IMAGE_HEIGHT,
                      task.stream); 
    check_cuda(cudaGetLastError(), "launch_preprocess");
    
    // 4. 将任务推入队列，让推理线程接管
    if (task_queue.push(std::move(task)) == tools::QueuePushResult::Closed) return false;
    in_flight_count++;

    // 5. 核心逻辑：流水线深度控制 (Pipeline Depth)
    // 假设池子大小为3，我们允许最多有 2 帧在 GPU 流水线里跑。
    // 如果达到了 2 帧，主线程就阻塞等待最早的一帧完成，以维持稳定的内存循环。
    const int MAX_IN_FLIGHT = 2; 
    
    if (in_flight_count >= MAX_IN_FLIGHT) {
        // 阻塞获取最早推入的帧的计算结果
        auto finished = result_queue.wait_pop();
        if (!finished.has_value()) return false;
        TRTFrameData finished_task = std::move(*finished);
        in_flight_count--;

        // 在 CPU 端执行 YOLO Sigmoid 和 NMS (纯 CPU 耗时操作)
        out_objects = decode_outputs(finished_task.output_host, finished_task.detect_color);
        
        // 切片保存旧帧的数据（包含了当时的 frame 和 gimbal_q）
        out_frame_data = finished_task; 

        // 结果已取出，将 Buffer 洗净后归还给空闲池
        free_buffer_queue.push(std::move(finished_task));
        return true; 
    }

    // 流水线还在预热阶段（刚启动的前 1~2 帧），暂无结果返回
    return false;
}

std::vector<Object> TensorrtInferEngine::decode_outputs(const float* output_ptr, int detect_color){


    objects.clear();
    tmp_objects.clear();

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
        const float* ptr = output_ptr + i * cols;

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
        obj.color = !color_id;
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

    auto t3 = std::chrono::steady_clock::now();

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for (int idx : indices) {
        if (idx < (int)objects.size()) tmp_objects.push_back(objects[idx]);
    }
    auto t4 = std::chrono::steady_clock::now();

    return tmp_objects;
}


void TensorrtInferEngine::infer_workerLoop(){
    // std::cout << "[Worker] Thread started" << std::endl;
    try {
        while (auto queued_task = task_queue.wait_pop()) {
            TRTFrameData task = std::move(*queued_task);
            execute(task);
            if (result_queue.push(std::move(task)) == tools::QueuePushResult::Closed) return;
        }
    } catch (const std::exception& error) {
        tools::logger()->error("[TensorRT 0708] worker failed: {}", error.what());
        task_queue.close();
        free_buffer_queue.close();
        result_queue.close();
    }
}

void TensorrtInferEngine::execute(TRTExecutionSlot<float>& slot) {
    if (!slot.context->setTensorAddress(input_name.c_str(), slot.input_device) ||
        !slot.context->setTensorAddress(output_name.c_str(), slot.output_device)) {
        throw std::runtime_error("TensorRT 0708 failed to bind inference tensors");
    }
    if (!slot.context->enqueueV3(slot.stream)) {
        throw std::runtime_error("TensorRT 0708 enqueueV3 failed");
    }
    check_cuda(
      cudaMemcpyAsync(slot.output_host, slot.output_device, output_size,
                      cudaMemcpyDeviceToHost, slot.stream),
      "cudaMemcpyAsync(output)");
    check_cuda(cudaStreamSynchronize(slot.stream), "cudaStreamSynchronize");
}


TensorRTYolo0708::TensorRTYolo0708(const std::string& config_path, bool debug)
    : debug_(debug)
{
    auto yaml = YAML::LoadFile(config_path);

    std::string engine_path = yaml["trt_engine_path_0708"].as<std::string>();
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

std::list<Armor> TensorRTYolo0708::detect(const cv::Mat& raw_img, int frame_count)
{
    if (raw_img.empty()) {
        tools::logger()->warn("Empty image in TensorRTYolo0708::detect");
        return {};
    }

    cv::Mat inference_img = raw_img;
    if (use_roi_) {
        try {
            inference_img = raw_img(resolve_roi(roi_, raw_img.size()));
        } catch (const std::exception& error) {
            tools::logger()->error("[TensorRTYolo0708] invalid ROI: {}", error.what());
            return {};
        }
    }

    // 调用 TensorRT 推理
    engine_->infer(inference_img, detect_color_);
    // 转换为 Armor 列表
    return convertToArmors(inference_img, raw_img, frame_count);
}

YOLOFrameData TensorRTYolo0708::detect(YOLOFrameData frame_data, int frame_count){
if (frame_data.frame.empty()) {
        tools::logger()->warn("Empty image in TensorRTYolo0708::detect");
        return YOLOFrameData(); // 返回空
    }

    cv::Mat inference_img = frame_data.frame;
    if (use_roi_) {
        try {
            inference_img = frame_data.frame(resolve_roi(roi_, frame_data.frame.size()));
        } catch (const std::exception& error) {
            tools::logger()->error("[TensorRTYolo0708] invalid ROI: {}", error.what());
            return YOLOFrameData();
        }
    }

    std::vector<Object> parsed_objects;
    YOLOFrameData finished_frame; // 存放几十毫秒前那张图的完整数据

    // 塞入最新帧，同时尝试获取历史帧结果
    bool has_result = engine_->async_enabled_infer(
      inference_img, frame_data, frame_count, detect_color_, parsed_objects, finished_frame);

    if (!has_result) {
        // 流水线预热中，直接返回空结果给外层，外层这帧不要进行自瞄解算

        YOLOFrameData empty;
        empty.is_empty = true;
        return empty; 
    }

    // 兼容之前的代码结构，更新类内部状态
    engine_->tmp_objects = parsed_objects;

    // 重点：生成 Armor 列表。
    // 必须传入 finished_frame 的图片和 frame_count，因为缩放比例(sx, sy)依赖于当时截图的大小
    cv::Mat finished_inference_img = finished_frame.frame;
    if (use_roi_) {
        finished_inference_img =
          finished_frame.frame(resolve_roi(roi_, finished_frame.frame.size()));
    }
    finished_frame.armors =
      convertToArmors(finished_inference_img, finished_frame.frame, frame_count);

    // 返回旧帧。外层代码将使用 finished_frame.gimbal_q 和 finished_frame.armors 进行 PNP 和坐标系转换
    finished_frame.is_empty = false;
    return finished_frame;
} 

std::list<Armor> TensorRTYolo0708::convertToArmors(
  const cv::Mat& inference_img, const cv::Mat& raw_img, int frame_count)
{
    std::list<Armor> armors;
    const auto& objects = engine_->tmp_objects;

    // 计算缩放因子
    float sx = static_cast<float>(inference_img.cols) / engine_->IMAGE_WIDTH;
    float sy = static_cast<float>(inference_img.rows) / engine_->IMAGE_HEIGHT;

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
        armor.center_norm = cv::Point2f(
          armor.center.x / raw_img.cols, armor.center.y / raw_img.rows);

        armors.push_back(armor);
    }

    if (debug_) drawDetections(raw_img, armors, frame_count);
    return armors;
}

void TensorRTYolo0708::sortKeypoints(std::vector<cv::Point2f>& pts)
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

bool TensorRTYolo0708::checkName(const Armor& armor) const
{
    return armor.name != ArmorName::not_armor;
}

bool TensorRTYolo0708::checkType(const Armor& armor) const
{
    if (armor.type == ArmorType::small) {
        return armor.name != ArmorName::one && armor.name != ArmorName::base;
    } else {
        return armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
               armor.name != ArmorName::outpost;
    }
}

ArmorType TensorRTYolo0708::getType(const Armor& armor) const
{
    if (armor.name == ArmorName::one || armor.name == ArmorName::base)
        return ArmorType::big;
    if (armor.name == ArmorName::two || armor.name == ArmorName::sentry ||
        armor.name == ArmorName::outpost)
        return ArmorType::small;
    return ArmorType::small;
}

void TensorRTYolo0708::drawDetections(const cv::Mat& img, const std::list<Armor>& armors,
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

std::list<Armor> TensorRTYolo0708::postprocess(double scale, cv::Mat& output,
                                           const cv::Mat& bgr_img, int frame_count)
{
    return {};  // TensorRT 这里不调用
}


} // namespace auto_aim
