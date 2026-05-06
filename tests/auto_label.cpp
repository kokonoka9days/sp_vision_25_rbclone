#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
// 确保这个头文件的路径与你的工程目录结构匹配
#include "tasks/auto_buff/yolo11_buff.hpp"

namespace fs = std::filesystem;

int main() {
    // 1. 初始化你的模型类
    auto_buff::YOLO11_BUFF detector("../configs/xiaohuang.yaml");

    // ==========================================
    // 2. 配置文件路径（请根据你的实际情况修改）
    // ==========================================
    // 你的输入视频路径
    std::string video_path = "/home/cyn/Desktop/sp_vision_25_rbclone/val_image.avi"; 
    
    // 分别设置图片和标签的独立输出路径
    std::string img_output_folder = "/home/cyn/Desktop/datasets/buff/images/val";  
    std::string txt_output_folder = "/home/cyn/Desktop/datasets/buff/labels/val";  
    
    // 如果输出文件夹不存在，则自动创建它们
    if (!fs::exists(img_output_folder)) {
        fs::create_directories(img_output_folder);
    }
    if (!fs::exists(txt_output_folder)) {
        fs::create_directories(txt_output_folder);
    }

    // 3. 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频文件: " << video_path << std::endl;
        return -1;
    }

    // 抽帧参数配置
    int step = 4;             // 抽帧步长：3 表示每 3 帧抽 1 帧
    int frame_counter = 0;    // 用于记录视频播放到了第几帧
    int saved_counter = 0;    // 用于记录实际保存了多少帧（用于命名，保证序号连续）
    cv::Mat frame;

    std::cout << "开始读取视频并进行抽帧标注..." << std::endl;

    // 4. 循环读取视频帧
    while (cap.read(frame)) {
        frame_counter++;

        // 如果不是设定的步长倍数，就跳过这一帧（实现抽帧）
        if (frame_counter % step != 0) {
            continue;
        }

        int img_w = frame.cols;
        int img_h = frame.rows;

        // 5. 调用推理函数跑模型
        auto results = detector.get_onecandidatebox(frame); 

        // 6. 只有当画面中识别到目标时，才保存图片和生成 TXT
        if (!results.empty()) {
            
            // 格式化文件名，例如：frame_00000, frame_00001
            char filename_buf[256];
            sprintf(filename_buf, "frame_%05d", saved_counter);
            std::string base_name = filename_buf;
            
            // 分别拼接图片和 TXT 的完整保存路径
            std::string img_path = (fs::path(img_output_folder) / (base_name + ".jpg")).string();
            std::string txt_path = (fs::path(txt_output_folder) / (base_name + ".txt")).string();

            // 【第一步】保存 .jpg 图片到 images/train
            cv::imwrite(img_path, frame);

            // 【第二步】生成 YOLO 格式的 .txt 标签到 labels/train
            std::ofstream out(txt_path);
            
            // 遍历识别到的所有扇叶目标
            for (size_t j = 0; j < results.size(); ++j) {
                const auto& obj = results[j];
                
                // YOLO 要求所有坐标必须归一化到 0.0 ~ 1.0 之间
                // 计算中心点 (cx, cy) 和 宽高 (w, h)
                float cx = (obj.rect.x + obj.rect.width / 2.0f) / (float)img_w;
                float cy = (obj.rect.y + obj.rect.height / 2.0f) / (float)img_h;
                float w = obj.rect.width / (float)img_w;
                float h = obj.rect.height / (float)img_h;

                // 写入: 类别ID(0) cx cy w h
                out << "0 " << cx << " " << cy << " " << w << " " << h;

                // 写入6个关键点的归一化坐标
                for (int p = 0; p < 6; ++p) {
                    float kpt_x = obj.kpt[p].x / (float)img_w;
                    float kpt_y = obj.kpt[p].y / (float)img_h;
                    out << " " << kpt_x << " " << kpt_y;
                }
                
                // 换行，准备写入画面中的下一个目标（如果有的话）
                out << "\n";
            }
            
            out.close();
            saved_counter++;
            std::cout << "✅ 成功截取并生成标签: " << base_name << " (已分别存入 images 和 labels 文件夹)" << std::endl;
        }
    }
    
    cap.release(); // 释放视频流
    std::cout << "🎉 视频处理完成！共生成 " << saved_counter << " 对包含目标的图片和标签。" << std::endl;
    return 0;
}