#include <fmt/core.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <opencv2/opencv.hpp>

// 引入 Ceres 求解器
#include <ceres/ceres.h> 

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"
#include "tools/logger.hpp"

using namespace std::chrono_literals;

struct SyncData {
    std::chrono::steady_clock::time_point t_cam;
    double vision_yaw; 
};

// ================= 新增：Ceres 优化的代价函数 =================
struct VarianceCostFunctor {
    VarianceCostFunctor(const std::vector<SyncData>& data, io::Gimbal* gimbal)
        : data_(data), gimbal_(gimbal) {}

    // Ceres 的 operator() 会传入当前待优化的参数 dt_ms
    bool operator()(const double* const dt_ms, double* residual) const {
       auto dt = std::chrono::nanoseconds(static_cast<int64_t>(dt_ms[0] * 1000000.0));
        std::vector<double> absolute_yaws;
        absolute_yaws.reserve(data_.size());

        for (const auto& d : data_) {
            try {
                auto q = gimbal_->q(d.t_cam + dt);
                auto ypr = tools::eulers(q, 2, 1, 0); 
                absolute_yaws.push_back(ypr[0] + d.vision_yaw);
            } catch (...) {
                continue;
            }
        }

        // 如果大部分数据因为超出队列找不到姿态，返回一个极大的残差作为惩罚
        if (absolute_yaws.size() < data_.size() * 0.8) {
            residual[0] = 1e6; 
            return true;
        }

        double sum = std::accumulate(absolute_yaws.begin(), absolute_yaws.end(), 0.0);
        double mean = sum / absolute_yaws.size();
        
        double variance = 0.0;
        for (double yaw : absolute_yaws) {
            variance += (yaw - mean) * (yaw - mean);
        }
        variance /= absolute_yaws.size();

        // Ceres 的目标是最小化 residual 的平方，我们这里返回标准差（方差的平方根）
        residual[0] = std::sqrt(variance);
        return true;
    }

    const std::vector<SyncData>& data_;
    io::Gimbal* gimbal_;
};
// ==========================================================

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | ../configs/xiaohei.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char* argv[]) {
    tools::Exiter exiter;
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    io::Gimbal gimbal(config_path);
    io::Camera camera(config_path);
    auto_aim::YOLO yolo(config_path, true);
    auto_aim::Solver solver(config_path); 

    std::vector<SyncData> calibration_data;
    bool is_recording = false;

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    tools::logger()->info("====== 时间同步校准工具 ======");
    // ... （省略前面的固定日志输出）

    while (!exiter.exit()) {
        camera.read(img, t);
        auto armors = yolo.detect(img);

        if (!armors.empty() && is_recording) {
            auto armor = armors.front();
            cv::Point2f center = (armor.left.top + armor.left.bottom + 
                      armor.right.top + armor.right.bottom) / 4.0f;
            
            double cx = img.cols / 2.0;
            double fx = 12.0; 
            double vision_yaw = std::atan((center.x - cx) / fx);

            calibration_data.push_back({t, vision_yaw});
            cv::circle(img, center, 10, cv::Scalar(0, 0, 255), -1);
        }

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
            tools::logger()->info("结束录制，共收集到 {} 帧数据。开始 Ceres 优化计算...", calibration_data.size());

            if (calibration_data.size() < 50) {
                tools::logger()->warn("数据量太少（<50帧），请重新录制！");
                continue;
            }

            // ================= 核心：使用 Ceres 替代暴力搜索 =================
            double best_dt_ms = 1.0; // 初始估计值

            ceres::Problem problem;
            // 使用数值求导构建代价函数 (CENTRAL中心差分, 1个残差, 1个参数)
            ceres::CostFunction* cost_function =
                new ceres::NumericDiffCostFunction<VarianceCostFunctor, ceres::CENTRAL, 1, 1>(
                    new VarianceCostFunctor(calibration_data, &gimbal));

            problem.AddResidualBlock(cost_function, nullptr, &best_dt_ms);
            
            // 设定时间补偿的合理上下界，防止优化时跑偏
            problem.SetParameterLowerBound(&best_dt_ms, 0, -10.0);
            problem.SetParameterUpperBound(&best_dt_ms, 0, 10.0);

            ceres::Solver::Options options;
            options.minimizer_progress_to_stdout = true; 
            options.max_num_iterations = 50;           
            options.linear_solver_type = ceres::DENSE_QR;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            tools::logger()->info("\n{}", summary.BriefReport());
            
            // 构造参数指针数组
            const double* parameters[1] = { &best_dt_ms };
            // 准备接收残差的变量
            double residual = 0.0;

            // 调用 Evaluate，结果写入 residual 中
            cost_function->Evaluate(parameters, &residual, nullptr);

            // 计算最终方差
            double final_variance = std::pow(residual, 2);

            tools::logger()->info("====== 计算结果 ======");
            tools::logger()->info("最佳时间补偿为:  {:.1f} ms", best_dt_ms);
            // tools::logger()->info("最小方差为:      {:.6f}", final_variance); 
            tools::logger()->info("建议在 rb_auto_aim_debug 中修改为: auto q = gimbal.q(t {} {:.1f}ms);", 
                                  best_dt_ms >= 0 ? "+" : "-", std::abs(best_dt_ms));
            tools::logger()->info("======================");
        }
    }

    return 0;
}