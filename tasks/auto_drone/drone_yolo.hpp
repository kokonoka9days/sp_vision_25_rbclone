#ifndef AUTO_DRONE__DRONE_YOLO_HPP
#define AUTO_DRONE__DRONE_YOLO_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
// 引入 OpenVINO 2022+ 标准头文件
#include <openvino/openvino.hpp> 

#include "drone_armor.hpp"

namespace auto_drone
{

class YOLO {
private:
    int input_w_;             
    int input_h_;             
    int num_classes_;         
    int num_kpts_;            
    int num_boxes_;           
    float score_threshold_;    
    float nms_threshold_;

    // OpenVINO 核心推理组件
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    // 仿射变换参数
    float scale_;                                       
    int pad_w_;                                           
    int pad_h_;

    std::vector<Drone> postprocessing(float* output);                 

public:
    YOLO(const std::string& config_path, bool debug = false);
    ~YOLO() = default; 

    std::vector<Drone> detect(const cv::Mat &frame);
};

} // namespace auto_drone

#endif // AUTO_DRONE__DRONE_YOLO_HPP