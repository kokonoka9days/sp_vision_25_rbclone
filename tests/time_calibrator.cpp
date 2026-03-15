#include <fmt/core.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"
#include "tools/logger.hpp"

using namespace std::chrono_literals;

// 存储时间戳与视觉偏航角的数据结构
struct SyncData {
    std::chrono::steady_clock::time_point t_cam;
    double vision_yaw; // 装甲板在相机坐标系下的偏航角 (弧度)
};

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | ../configs/sb_copy.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char* argv[]) {
    tools::Exiter exiter;
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    // 初始化硬件和算法
    io::Gimbal gimbal(config_path);
    io::Camera camera(config_path);
    auto_aim::YOLO yolo(config_path, true);
    auto_aim::Solver solver(config_path); // 用于获取相机的内参或执行PnP

    std::vector<SyncData> calibration_data;
    bool is_recording = false;

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    tools::logger()->info("====== 时间同步校准工具 ======");
    tools::logger()->info("操作指南:");
    tools::logger()->info("1. 将一个装甲板固定在相机正前方静止不动。");
    tools::logger()->info("2. 按下 'c' 键开始录制数据，此时请左右快速晃动云台（持续2-3秒）。");
    tools::logger()->info("3. 按下 'x' 键停止录制并自动计算最佳时间补偿（精度100us）。");
    tools::logger()->info("4. 按下 'q' 键退出程序。");
    tools::logger()->info("===============================");

    while (!exiter.exit()) {
        camera.read(img, t);
        
        // 检测装甲板 (不使用Tracker，避免EKF滤波带来的相位延迟)
        auto armors = yolo.detect(img);

        if (!armors.empty() && is_recording) {
            // 取画面中最大的或者第一个装甲板
            auto armor = armors.front();
            
            // 【关键】计算视觉偏航角 (Vision Yaw)
            // 这里我们需要装甲板相对于相机中心的角度。
            // 简单近似法：使用2D图像中心的水平像素偏差与相机内参焦距(fx)的比例。
            // 假设 solver.K() 能够获取相机内参矩阵：
            // 如果无法获取内参，可以用近似值，只要线性度足够就能找到相位差
          cv::Point2f center = (armor.left.top + armor.left.bottom + 
                      armor.right.top + armor.right.bottom) / 4.0f;
            
            // 假设相机中心在图像正中，简单粗略估算视场角偏角 (此处以像素偏差暂代，求极值点时相位特征是一致的)
            // 更好的做法是调用你的 PnP 解算获取 Camera 系下的 X 和 Z，计算 atan2(X, Z)
            double cx = img.cols / 2.0;
            double fx = 12.0; // 替换为你的真实焦距，若只是测相位，这个值影响不大
            double vision_yaw = std::atan((center.x - cx) / fx);

            calibration_data.push_back({t, vision_yaw});
            
            // 画面上打个标记表示正在录入该装甲板
            cv::circle(img, center, 10, cv::Scalar(0, 0, 255), -1);
        }

        // 显示状态
        std::string state_text = is_recording ? fmt::format("Recording... Data size: {}", calibration_data.size()) : "Waiting... Press 'c' to start";
        cv::putText(img, state_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                    is_recording ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);

        cv::imshow("Time Calibration", img);
        auto key = cv::waitKey(1);

        if (key == 'q') {
            break;
        } 
        else if (key == 'c' && !is_recording) {
            calibration_data.clear();
            is_recording = true;
            tools::logger()->info("开始录制数据，请左右晃动云台...");
        } 
        else if (key == 'x' && is_recording) {
            is_recording = false;
            tools::logger()->info("结束录制，共收集到 {} 帧数据。开始计算...", calibration_data.size());

            if (calibration_data.size() < 50) {
                tools::logger()->warn("数据量太少（<50帧），请重新录制！");
                continue;
            }

            // === 核心算法：暴力搜索 100us 精度的最佳时间补偿 ===
            double best_dt_ms = 0.0;
            double min_variance = std::numeric_limits<double>::max();

            // 搜索范围：-10.0ms 到 10.0ms，步长 0.1ms (100us)
            for (double dt_ms = -10.0; dt_ms <= 10.0; dt_ms += 0.1) {
                auto dt = std::chrono::microseconds(static_cast<int>(dt_ms * 1000));
                
                std::vector<double> absolute_yaws;
                absolute_yaws.reserve(calibration_data.size());

                for (const auto& data : calibration_data) {
                    try {
                        // 尝试获取加上时间偏移 dt 后的云台姿态
                        auto q = gimbal.q(data.t_cam + dt);
                        auto ypr = tools::eulers(q, 2, 1, 0); // Yaw, Pitch, Roll
                        double gimbal_yaw = ypr[0];
                        
                        // 绝对偏航角 = 云台偏航角 + 视觉偏航角
                        absolute_yaws.push_back(gimbal_yaw + data.vision_yaw);
                    } catch (...) {
                        // 忽略查不到时间戳的数据点（超出了队列缓存等）
                        continue;
                    }
                }

                // 只有当大部分数据都能查到姿态时才进行统计
                if (absolute_yaws.size() > calibration_data.size() * 0.8) {
                    double sum = std::accumulate(absolute_yaws.begin(), absolute_yaws.end(), 0.0);
                    double mean = sum / absolute_yaws.size();
                    
                    double variance = 0.0;
                    for (double yaw : absolute_yaws) {
                        variance += (yaw - mean) * (yaw - mean);
                    }
                    variance /= absolute_yaws.size();

                    if (variance < min_variance) {
                        min_variance = variance;
                        best_dt_ms = dt_ms;
                    }
                }
            }

            tools::logger()->info("====== 计算结果 ======");
            tools::logger()->info("最佳时间补偿为:  {:.1f} ms", best_dt_ms);
            tools::logger()->info("最小方差为:      {:.6f}", min_variance);
            tools::logger()->info("建议在 rb_auto_aim_debug 中修改为: auto q = gimbal.q(t {} {:.1f}ms);", 
                                  best_dt_ms >= 0 ? "+" : "-", std::abs(best_dt_ms));
            tools::logger()->info("======================");
        }
    }

    return 0;
}