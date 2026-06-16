// Calibration pipeline: camera intrinsics -> robot-world-hand-eye.
// Camera intrinsics auto-feed into hand-eye; no manual YAML copy needed.
// Requires images + quaternions already captured in ../assets/img_with_q/.

#include <fmt/core.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace fs = std::filesystem;

static const std::string CONFIG_PATH = "../configs/calibration.yaml";
static const fs::path    OUTPUT_DIR  = "../assets/img_with_q";

static int    g_pattern_cols = 11;
static int    g_pattern_rows = 8;
static double g_square_mm    = 30.0;
static cv::Size g_img_size;

// ============================================================================
// Helpers
// ============================================================================

static bool read_q(const fs::path& path, Eigen::Quaterniond& q) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    double w, x, y, z;
    ifs >> w >> x >> y >> z;
    q = Eigen::Quaterniond(w, x, y, z);
    return true;
}

// ============================================================================
// 3-D board points — two conventions
// ============================================================================

// calibrate_camera convention: Z=0 plane
static std::vector<cv::Point3f> chessboard_3d_camera() {
    std::vector<cv::Point3f> pts;
    for (int i = 0; i < g_pattern_rows; ++i)
        for (int j = 0; j < g_pattern_cols; ++j)
            pts.emplace_back(j * g_square_mm, i * g_square_mm, 0.0f);
    return pts;
}

// calibrate_robotworld_handeye convention: X=0 plane (Y-Z)
static std::vector<cv::Point3f> chessboard_3d_handeye() {
    std::vector<cv::Point3f> pts;
    for (int i = 0; i < g_pattern_rows; ++i)
        for (int j = 0; j < g_pattern_cols; ++j)
            pts.emplace_back(0.0f, j * g_square_mm, i * g_square_mm);
    return pts;
}

// ============================================================================
// Camera intrinsics calibration
// ============================================================================

static void camera_load(std::vector<std::vector<cv::Point3f>>& obj_pts,
                        std::vector<std::vector<cv::Point2f>>& img_pts) {
    const auto corners_3d = chessboard_3d_camera();
    const cv::Size pattern(g_pattern_cols, g_pattern_rows);

    for (int i = 1; ; ++i) {
        auto path = OUTPUT_DIR / fmt::format("{}.jpg", i);
        cv::Mat img = cv::imread(path);
        if (img.empty()) break;

        g_img_size = img.size();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, pattern, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            obj_pts.push_back(corners_3d);
            img_pts.push_back(corners);
            cv::drawChessboardCorners(img, pattern, corners, true);
        }

        std::string status = found ? "OK" : "FAIL";
        cv::putText(img, "img " + std::to_string(i) + ": " + status,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("Camera calibration", img);
        cv::waitKey(found ? 500 : 0);
    }
    cv::destroyWindow("Camera calibration");
}

static void camera_print_yaml(const cv::Mat& camera_matrix,
                              const cv::Mat& distort_coeffs,
                              double reproj_err) {
    fmt::print("\n# --- Camera intrinsics result ---\n");
    fmt::print("camera_matrix: [");
    for (int i = 0; i < 9; ++i)
        fmt::print("{:.3f}{}", camera_matrix.at<double>(i), (i == 8) ? "" : ", ");
    fmt::print("]\n");

    fmt::print("distort_coeffs: [");
    for (int i = 0; i < distort_coeffs.cols; ++i)
        fmt::print("{:.6f}{}", distort_coeffs.at<double>(i),
                   (i == distort_coeffs.cols - 1) ? "" : ", ");
    fmt::print("]\n");
    fmt::print("# reprojection error: {:.3f} px\n\n", reproj_err);
}

// ============================================================================
// Robot-world-hand-eye calibration
// ============================================================================

struct HandEyeData {
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<cv::Mat> R_w2g, t_w2g;
    cv::Mat R_w2b, t_w2b;
    cv::Mat R_g2c, t_g2c;
};

static void handeye_load(HandEyeData& data,
                         const cv::Mat& camera_matrix,
                         const cv::Mat& distort_coeffs,
                         const cv::Mat& R_gimbal2imu) {
    const auto corners_3d = chessboard_3d_handeye();
    const cv::Size pattern(g_pattern_cols, g_pattern_rows);

    for (int i = 1; ; ++i) {
        auto img_path = OUTPUT_DIR / fmt::format("{}.jpg", i);
        auto   q_path = OUTPUT_DIR / fmt::format("{}.txt", i);
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) break;

        Eigen::Quaterniond q;
        if (!read_q(q_path, q)) {
            tools::logger()->warn("missing {}", q_path.string());
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        if (!cv::findChessboardCorners(gray, pattern, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE))
            continue;

        cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

        // solvePnP: board pose in camera frame
        cv::Mat rvec, tvec;
        cv::solvePnP(corners_3d, corners, camera_matrix, distort_coeffs, rvec, tvec);

        // world -> gimbal from IMU quaternion: R_w2g = R_w2imu * R_g2imu^T
        Eigen::Matrix3d R_w2imu = q.matrix();
        Eigen::Matrix3d R_g2imu_eigen;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_g2imu_eigen(r, c) = R_gimbal2imu.at<double>(r, c);
        Eigen::Matrix3d R_w2g_eigen = R_w2imu * R_g2imu_eigen.transpose();

        cv::Mat R_w2g(3, 3, CV_64F);
        cv::Mat t_w2g = cv::Mat::zeros(3, 1, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R_w2g.at<double>(r, c) = R_w2g_eigen(r, c);

        data.rvecs.push_back(rvec);
        data.tvecs.push_back(tvec);
        data.R_w2g.push_back(R_w2g);
        data.t_w2g.push_back(t_w2g);
    }
}

static void handeye_print_yaml(const HandEyeData& data,
                               const cv::Mat& camera_matrix,
                               const std::vector<double>& R_gimbal2imubody_data) {
    cv::Mat R_cam2gimbal = data.R_g2c.t();
    cv::Mat t_cam2gimbal = -R_cam2gimbal * (data.t_g2c / 1e3);  // mm -> m

    // Compute board pose in world frame
    cv::Mat R_board2world = data.R_w2b.t();
    cv::Mat t_board2world = -R_board2world * (data.t_w2b / 1e3);  // mm -> m

    // 相机同理想情况的偏角
    Eigen::Matrix3d R_c2g_eigen;
    cv::cv2eigen(R_cam2gimbal, R_c2g_eigen);
    Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
    Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_c2g_eigen;
    Eigen::Vector3d camera_ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;

    // 标定板到世界坐标系原点的水平距离
    double x = t_board2world.at<double>(0);
    double y = t_board2world.at<double>(1);
    double dist = std::sqrt(x * x + y * y);

    // 标定板同竖直摆放时的偏角
    Eigen::Matrix3d R_b2w_eigen;
    cv::cv2eigen(R_board2world, R_b2w_eigen);
    Eigen::Vector3d board_ypr = tools::eulers(R_b2w_eigen, 2, 1, 0) * 57.3;

    // Output
    fmt::print("\n# --- Hand-eye result ---\n");

    fmt::print("R_gimbal2imubody: [");
    for (size_t i = 0; i < R_gimbal2imubody_data.size(); ++i)
        fmt::print("{}", R_gimbal2imubody_data[i],
                   (i == R_gimbal2imubody_data.size() - 1) ? "" : ", ");
    fmt::print("]\n\n");

    fmt::print("# 相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree\n",
               camera_ypr[0], camera_ypr[1], camera_ypr[2]);
    fmt::print("# 标定板到世界坐标系原点的水平距离: {:.2f} m\n", dist);
    fmt::print("# 标定板同竖直摆放时的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree\n",
               board_ypr[0], board_ypr[1], board_ypr[2]);
    fmt::print("\n");

    fmt::print("R_camera2gimbal: [");
    for (int i = 0; i < 9; ++i)
        fmt::print("{:.16f}{}", R_cam2gimbal.at<double>(i), (i == 8) ? "" : ", ");
    fmt::print("]\n");

    fmt::print("t_camera2gimbal: [");
    for (int i = 0; i < 3; ++i)
        fmt::print("{:.16f}{}", t_cam2gimbal.at<double>(i), (i == 2) ? "" : ", ");
    fmt::print("]\n\n");
}

// ============================================================================
// Main pipeline
// ============================================================================

int main() {
    // Load config
    auto cfg = tools::load(CONFIG_PATH);
    g_pattern_cols = cfg["pattern_cols"].as<int>();
    g_pattern_rows = cfg["pattern_rows"].as<int>();
    g_square_mm    = cfg["square_size_mm"].as<double>();
    auto R_gimbal2imubody_data = cfg["R_gimbal2imubody"].as<std::vector<double>>();
    cv::Mat R_gimbal2imu(3, 3, CV_64F, R_gimbal2imubody_data.data());
    R_gimbal2imu = R_gimbal2imu.clone();

    // ---- Step 1: Camera intrinsics ----
    tools::logger()->info("=== Camera calibration ===");

    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    camera_load(obj_pts, img_pts);

    if (obj_pts.size() < 3) {
        tools::logger()->error("Need >= 3 valid images, got {}", obj_pts.size());
        return 1;
    }

    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::calibrateCamera(obj_pts, img_pts, g_img_size, camera_matrix, distort_coeffs, rvecs, tvecs);

    double total_err = 0;
    int total_pts = 0;
    for (size_t k = 0; k < obj_pts.size(); ++k) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(obj_pts[k], rvecs[k], tvecs[k], camera_matrix, distort_coeffs, projected);
        for (size_t j = 0; j < projected.size(); ++j) {
            double dx = projected[j].x - img_pts[k][j].x;
            double dy = projected[j].y - img_pts[k][j].y;
            total_err += std::sqrt(dx * dx + dy * dy);
            ++total_pts;
        }
    }
    camera_print_yaml(camera_matrix, distort_coeffs, total_err / total_pts);

    // ---- Step 2: Hand-eye (intrinsics auto-fed) ----
    tools::logger()->info("=== Hand-eye calibration ===");

    HandEyeData he_data;
    handeye_load(he_data, camera_matrix, distort_coeffs, R_gimbal2imu);
    tools::logger()->info("loaded {} samples", he_data.rvecs.size());

    if (he_data.rvecs.size() < 2) {
        tools::logger()->error("Need >= 2 valid samples, got {}", he_data.rvecs.size());
        return 1;
    }

    cv::calibrateRobotWorldHandEye(he_data.rvecs, he_data.tvecs,
                                   he_data.R_w2g, he_data.t_w2g,
                                   he_data.R_w2b, he_data.t_w2b,
                                   he_data.R_g2c, he_data.t_g2c);

    handeye_print_yaml(he_data, camera_matrix, R_gimbal2imubody_data);
    return 0;
}
