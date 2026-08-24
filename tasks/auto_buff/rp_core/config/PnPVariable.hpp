#pragma once
#include <opencv2/opencv.hpp>
#include "json.hpp"

inline cv::Matx<double, 3, 3> CAM;
inline cv::Matx<double, 1, 5> DIS;

inline void update_power_rune_camera_calibration()
{
    CAM = J_POWER_RUNE.camera_matrix();
    DIS = J_POWER_RUNE.distortion();
}
