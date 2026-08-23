#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#include "../ov_yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  /** @brief 初始化多线程 OpenVINO 检测器 @param config_path YAML 配置文件路径 @param debug 是否启用调试输出 */
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  /** @brief 提交一帧图像进行异步检测 @param img 输入图像 @param t 采集时间戳 */
  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  /** @brief 阻塞获取检测结果 @return 装甲板列表和原始时间戳 */
  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  /** @brief 阻塞获取包含原图的调试检测结果 @return 原图、装甲板列表和时间戳 */
  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  YOLO yolo_;

  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
