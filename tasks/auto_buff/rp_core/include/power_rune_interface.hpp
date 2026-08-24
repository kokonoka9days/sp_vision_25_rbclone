#ifndef AUTO_BUFF__RP_CORE__INTERFACE_HPP
#define AUTO_BUFF__RP_CORE__INTERFACE_HPP

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "runtime_types.hpp"

namespace power_rune
{
struct RuneInput
{
  struct NNRuneInfo
  {
    cv::Point top;
    cv::Point left;
    cv::Point right;
    cv::Point bottom;
    cv::Point point_R;
    int class_id = 0;
  };

  bool is_big_rune = false;
  cv::Mat ori_mat;
  CameraPose camera_pose;
  RuneTimestamp timestamp{};
  int target_color = 0;
  std::vector<NNRuneInfo> nn_rune_infos;
};

struct RuneSendData
{
  float yaw = 0.0f;
  float pitch = 0.0f;
  uint8_t is_find_buff = 0;
  uint8_t mode = 0;
  uint8_t is_enable_fire = 0;
};
}  // namespace power_rune

#endif
