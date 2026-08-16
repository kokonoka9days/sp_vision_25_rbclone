#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <memory>
#include <opencv2/opencv.hpp>

namespace auto_buff
{
inline const std::vector<std::string> class_names = {
  "inactive_target", "inactive_fan", "rune_center"};

enum RuneClass : int
{
  INACTIVE_TARGET = 0,
  INACTIVE_FAN = 1,
  RUNE_CENTER = 2
};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
    std::vector<float> kpt_conf;
  };

  /** @brief 根据配置加载能量机关 YOLO11 模型 @param config YAML 配置路径 */
  YOLO11_BUFF(const std::string & config);
  /** @brief 释放推理后端 */
  ~YOLO11_BUFF();

  /** @brief 禁止复制构造 */
  YOLO11_BUFF(const YOLO11_BUFF &) = delete;
  /** @brief 禁止复制赋值 */
  YOLO11_BUFF & operator=(const YOLO11_BUFF &) = delete;
  /** @brief 禁止移动构造 */
  YOLO11_BUFF(YOLO11_BUFF &&) = delete;
  /** @brief 禁止移动赋值 */
  YOLO11_BUFF & operator=(YOLO11_BUFF &&) = delete;

  /** @brief 推理并通过 NMS 返回多个候选框 @param image 输入及调试绘制图像 @return 检测对象列表 */
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  /** @brief 推理并返回最高置信候选框 @param image 输入及调试绘制图像 @return 每类最高置信对象 */
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  struct Backend;
  std::unique_ptr<Backend> backend_;

  static constexpr int NUM_CLASSES = 3;
  static constexpr int NUM_POINTS = 4;
  static constexpr int KPT_DIMS = 3;

  float confidence_threshold_ = 0.7f;
  float keypoint_threshold_ = 0.3f;
  float iou_threshold_ = 0.4f;

  /** @brief 执行推理并解码结果 @param image 输入图像 @return 检测对象列表 */
  std::vector<Object> infer_and_decode(cv::Mat & image);
  /** @brief 解码网络输出 @param output 输出数组 @param output_rows 行数 @param output_cols 列数 @param inverse_scale 逆缩放系数 @param image_size 原图尺寸 @return 候选对象列表 */
  std::vector<Object> decode(
    const float * output, int output_rows, int output_cols, float inverse_scale,
    const cv::Size & image_size) const;
  /** @brief 绘制检测对象和耗时 @param image 输出图像 @param objects 检测对象 @param elapsed_s 推理耗时，单位 s */
  void draw_objects(cv::Mat & image, const std::vector<Object> & objects, double elapsed_s) const;
};
}  // namespace auto_buff
#endif
