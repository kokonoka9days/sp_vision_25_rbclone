#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <chrono>

#include <opencv2/opencv.hpp>
#include <thread>

#include "armor.hpp"
#include "armor_interfaces.hpp"

namespace auto_aim
{
struct YOLOFrameData
{
  int detect_color = 0;     
  std::list<Armor> armors;
  Eigen::Quaterniond gimbal_q;
  std::chrono::steady_clock::time_point timestamp;
  cv::Mat frame;
  bool is_empty ;

  /** @brief 构造空帧数据 */
  YOLOFrameData():is_empty(true){};

  /** @brief 构造待检测帧数据 @param frame_ 图像 @param gimbal_q_ 采集时云台姿态 @param timestamp_ 采集时间戳 */
  YOLOFrameData(
    cv::Mat frame_, Eigen::Quaterniond gimbal_q_ = Eigen::Quaterniond(1,1,1,1),
    std::chrono::steady_clock::time_point timestamp_ = {})
  : gimbal_q(gimbal_q_), timestamp(timestamp_), frame(frame_), is_empty(false){};
};

class YOLOBase : public IArmorDetector
{
public:
  /** @brief 销毁 YOLO 后端 */
  virtual ~YOLOBase() = default;
  /** @brief 检测图像中的装甲板 @param img 输入图像 @param frame_count 帧编号 @return 装甲板列表 */
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) override = 0;
  /** @brief 检测带姿态和时间戳的帧 @param frame_data 输入帧数据 @param frame_count 帧编号 @return 填充检测结果后的帧数据 */
  virtual YOLOFrameData detect(YOLOFrameData frame_data, int frame_count = -1) = 0;
  /** @brief 将网络输出转换为装甲板 @param scale 网络输入缩放系数 @param output 网络输出张量 @param bgr_img 原始图像 @param frame_count 帧编号 @return 装甲板列表 */
  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

class YOLO : public IArmorDetector
{
public:
  /** @brief 根据配置创建指定 YOLO 后端 @param config_path YAML 配置文件路径 @param debug 是否启用调试输出 */
  YOLO(const std::string & config_path, bool debug = true);

  /** @brief 检测图像中的装甲板 @param img 输入图像 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1) override;

  /** @brief 检测带姿态和时间戳的帧 @param frame_data 输入帧数据 @param frame_count 帧编号 @return 填充检测结果后的帧数据 */
  YOLOFrameData detect(YOLOFrameData frame_data, int frame_count = -1);

  /** @brief 将网络输出转换为装甲板 @param scale 网络输入缩放系数 @param output 网络输出张量 @param bgr_img 原始图像 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
