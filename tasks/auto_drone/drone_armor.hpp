#ifndef AUTO_DRONE__DRONE_ARMOR_HPP
#define AUTO_DRONE__DRONE_ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_drone
{
// 颜色枚举保留
enum Color
{
  red,
  blue,
  extinguish,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

enum DroneName
{
  drone,
  not_drone
};
const std::vector<std::string> DRONE_NAMES = {"drone", "not_drone"};

struct Drone
{
  Color color;
  cv::Point2f center;       
  cv::Point2f center_norm;  
  std::vector<cv::Point2f> points; 

  DroneName name;
  int class_id;
  cv::Rect box;             
  double confidence;        

  Eigen::Vector3d xyz_in_gimbal;  
  Eigen::Vector3d xyz_in_world;   
  Eigen::Vector3d ypr_in_gimbal;  
  Eigen::Vector3d ypr_in_world;   
  Eigen::Vector3d ypd_in_world;   

  Drone() = default;

  Drone(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> drone_keypoints);
    
  Drone(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> drone_keypoints,
    cv::Point2f offset);
};

}  // namespace auto_drone

#endif  // AUTO_DRONE__DRONE_ARMOR_HPP