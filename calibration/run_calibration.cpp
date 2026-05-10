#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <fmt/core.h>

// 包含你的底层库
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp" 
#include "tools/math_tools.hpp"
#include "tools/logger.hpp"
#include "tools/img_tools.hpp"

// 包含你的算法库
#include "tasks/auto_drone/drone_yolo.hpp"
#include "tasks/auto_drone/drone_solver.hpp"
#include "tasks/auto_drone/drone_tracker.hpp"

using namespace std::chrono_literals;

// 1. 定义存储每帧标定数据的结构体
struct CalibData {
    double x, y, z;          // 视觉解算出的目标坐标
    double yaw_hit, pitch_hit; // 命中时云台的真实反馈角度 (弧度)
};

// 2. 角度归一化函数，防止跨越 180 度时产生误差
inline double wrap_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// 3. 高斯-牛顿非线性最小二乘求解器
void solve_calibration(const std::vector<CalibData>& data) {
    if (data.size() < 5) {
        fmt::print("[Warning] 数据点太少 ({} < 5)，拟合可能不准！\n", data.size());
    }

    // 待优化的参数矩阵 [dx, dy, dz, dYaw, dPitch]
    Eigen::VectorXd params(5);
    params.setZero(); // 初始猜测全为0

    int max_iter = 50;
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd J(2 * data.size(), 5); // 雅可比矩阵
        Eigen::VectorXd r(2 * data.size());    // 残差向量

        for (size_t i = 0; i < data.size(); ++i) {
            double dx = params[0], dy = params[1], dz = params[2];
            double dyaw = params[3], dpitch = params[4];

            double X = data[i].x + dx;
            double Y = data[i].y + dy;
            double Z = data[i].z + dz;

            double R2 = X * X + Y * Y;
            double R = std::sqrt(R2);
            double D2 = R2 + Z * Z;

            // 预测的角度
            double pred_yaw = std::atan2(Y, X) + dyaw;
            double pred_pitch = std::atan2(Z, R) + dpitch;

            // 计算残差
            r(2 * i) = wrap_angle(pred_yaw - data[i].yaw_hit);
            r(2 * i + 1) = wrap_angle(pred_pitch - data[i].pitch_hit);

            // Yaw 对各个参数的偏导数 (Jacobian)
            J(2 * i, 0) = -Y / R2;         // d(Yaw)/dx
            J(2 * i, 1) = X / R2;          // d(Yaw)/dy
            J(2 * i, 2) = 0.0;             // d(Yaw)/dz
            J(2 * i, 3) = 1.0;             // d(Yaw)/dYaw
            J(2 * i, 4) = 0.0;             // d(Yaw)/dPitch

            // Pitch 对各个参数的偏导数 (Jacobian)
            J(2 * i + 1, 0) = -(Z * X) / (D2 * R); // d(Pitch)/dx
            J(2 * i + 1, 1) = -(Z * Y) / (D2 * R); // d(Pitch)/dy
            J(2 * i + 1, 2) = R / D2;              // d(Pitch)/dz
            J(2 * i + 1, 3) = 0.0;                 // d(Pitch)/dYaw
            J(2 * i + 1, 4) = 1.0;                 // d(Pitch)/dPitch
        }

        // 高斯-牛顿迭代公式: delta = -(J^T * J)^-1 * J^T * r
        Eigen::VectorXd delta = (J.transpose() * J).ldlt().solve(-J.transpose() * r);
        params += delta;

        // 如果参数变化极小，说明已经收敛
        if (delta.norm() < 1e-6) break;
    }

    // 4. 打印结果，直接适配你的 auto_drone.yaml 格式
    fmt::print("\n============================================\n");
    fmt::print("标定完成！请将以下参数更新到 auto_drone.yaml :\n");
    fmt::print("xyz_offset: [{:.4f}, {:.4f}, {:.4f}]\n", params[0], params[1], params[2]);
    fmt::print("yaw_offset: {:.2f}\n", params[3] * 180.0 / M_PI);   // 弧度转度
    fmt::print("pitch_offset: {:.2f}\n", params[4] * 180.0 / M_PI); // 弧度转度
    fmt::print("============================================\n\n");
}

int main(int argc, char * argv[]) {
    std::string config_path = "../configs/auto_drone.yaml"; // 请根据实际情况修改
    if (argc > 1) config_path = argv[1];

    // 初始化硬件与算法
    io::Gimbal gimbal(config_path);
    io::Camera camera(config_path);
    auto_drone::YOLO yolo(config_path, true);
    auto_drone::Solver solver(config_path);
    auto_drone::Tracker tracker(config_path, &solver);
    tracker.set_gimbal(&gimbal);

    std::vector<CalibData> collected_data;

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    fmt::print("\n[标定程序启动]\n");
    fmt::print("操作说明:\n");
    fmt::print("  1. 遥控云台使激光完美命中目标\n");
    fmt::print("  2. 按下 's' 键保存当前帧数据 (建议在不同距离下保存 5-10 组)\n");
    fmt::print("  3. 收集完毕后，按下 'q' 键计算偏置并退出\n\n");

    while (true) {
        camera.read(img, t);
        auto ypr = gimbal.ypr(t);
        solver.set_R_gimbal2world(ypr[0], ypr[1], ypr[2]);

        auto drones = yolo.detect(img);
        auto targets = tracker.track(drones, t);

        // 获取云台真实反馈 (注意：需确认你的 gimbal.state().yaw 是角度还是弧度，这里假定是角度并转为弧度)
        auto gs = gimbal.state();
        double current_yaw_rad = gs.yaw * M_PI / 180.0;   
        double current_pitch_rad = gs.pitch * M_PI / 180.0;

        bool has_target = !targets.empty();
        Eigen::Vector3d xyz;
        if (has_target) {
            xyz = targets.front().get_xyz();
        }

        // --- 画面渲染 ---
        tools::draw_text(img, fmt::format("Collected Points: {}", collected_data.size()), {40, 40}, {0, 255, 0});
        if (has_target) {
            tools::draw_text(img, fmt::format("Target XYZ: {:.2f}, {:.2f}, {:.2f}", xyz.x(), xyz.y(), xyz.z()), {40, 80}, {0, 255, 255});
            
            // 画个中心十字准星辅助瞄准
            cv::Point center(img.cols / 2, img.rows / 2);
            cv::line(img, {center.x - 20, center.y}, {center.x + 20, center.y}, {0, 0, 255}, 2);
            cv::line(img, {center.x, center.y - 20}, {center.x, center.y + 20}, {0, 0, 255}, 2);
        } else {
            tools::draw_text(img, "No Target Detected!", {40, 80}, {0, 0, 255});
        }

        cv::resize(img, img, {}, 0.5, 0.5);  
        cv::imshow("Calibration Tool", img);

        // --- 按键响应 ---
        auto key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        } 
        else if (key == 's') {
            if (has_target) {
                collected_data.push_back({
                    xyz.x(), xyz.y(), xyz.z(),
                    current_yaw_rad, current_pitch_rad
                });
                fmt::print("已保存第 {} 组数据 | XYZ: [{:.2f}, {:.2f}, {:.2f}] | Gimbal: [{:.2f}°, {:.2f}°]\n", 
                           collected_data.size(), xyz.x(), xyz.y(), xyz.z(), gs.yaw, gs.pitch);
            } else {
                fmt::print("[警告] 画面中未检测到目标，无法保存！\n");
            }
        }
    }

    // 退出循环后执行高斯牛顿拟合
    if (!collected_data.empty()) {
        solve_calibration(collected_data);
    } else {
        fmt::print("未采集任何数据，程序退出。\n");
    }

    return 0;
}