#ifndef AUTO_DRONE__DRONE_YOLO_HPP
#define AUTO_DRONE__DRONE_YOLO_HPP

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "drone_armor.hpp"

namespace auto_drone
{

class YOLO {
private:
    // 模型基础参数
    int input_w_;             
    int input_h_;             
    int num_classes_;         // 动态类别数
    int num_kpts_;            // 动态关键点数 (无人机为 8 个)
    int num_boxes_;           // 输出框的数量 (例如 8400)
    float score_threshold_;    
    float nms_threshold_;
    bool use_cuda_preproc_;  

    // TensorRT 核心组件
    nvinfer1::IRuntime* runtime_ = nullptr;             
    nvinfer1::ICudaEngine* engine_ = nullptr;            
    nvinfer1::IExecutionContext* context_ = nullptr;     
    cudaStream_t stream_ = nullptr;

    // 显存指针 (Device)
    void* buffer_idx_0_ = nullptr; 
    void* buffer_idx_1_ = nullptr; 

    // CUDA 预处理图床
    uint8_t* d_src_img_ = nullptr; 
    int max_src_size_ = 0;

    // 锁页内存 (Host Pinned Memory - DMA 极速通道)
    float* pinned_in_host_ = nullptr;   
    float* pinned_out_host_ = nullptr;  
    int output_size_ = 0;              

    // 仿射变换参数 (用于坐标还原)
    float scale_;                                       
    int pad_w_;                                           
    int pad_h_;

    // 内部方法
    void preprocess_cuda(const cv::Mat &frame);  
    std::vector<Drone> postprocessing();                 

public:
    // 构造函数，兼容 rb_auto_drone_debug.cpp 的调用方式
    YOLO(const std::string& config_path, bool debug = false);
    ~YOLO();

    std::vector<Drone> detect(const cv::Mat &frame);
};

} // namespace auto_drone

#endif // AUTO_DRONE__DRONE_YOLO_HPP