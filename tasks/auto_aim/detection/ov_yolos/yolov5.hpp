#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <array>
#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "async_pipeline.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLOV5 : public YOLOBase
{
public:
  /** @brief 初始化 OpenVINO YOLOv5 检测器 @param config_path YAML 配置路径 @param debug 是否启用调试输出 */
  YOLOV5(const std::string & config_path, bool debug);

  /** @brief 同步检测图像中的装甲板 @param bgr_img BGR 图像 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /** @brief 异步流水线检测帧 @param frame_data 帧数据 @param frame_count 帧编号 @return 最早完成的帧结果 */
  YOLOFrameData detect(YOLOFrameData frame_data, int frame_count) override;
  /** @brief 后处理网络输出 @param scale 输入缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  static constexpr int kInputWidth = 640;
  static constexpr int kInputHeight = 640;
  static constexpr int kOutputValues = 22;
  static constexpr int kColorCount = 4;
  static constexpr int kClassCount = 9;

  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  float nms_threshold_;
  float score_threshold_;
  double min_confidence_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest sync_infer_request_;
  OpenVINOAsyncPipeline async_pipeline_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;
  friend class MultiThreadDetector;

  /** @brief 检查类别名称是否合法 @param armor 装甲板 @return 合法时返回 true */
  bool check_name(const Armor & armor) const;
  /** @brief 检查尺寸类型是否合法 @param armor 装甲板 @return 合法时返回 true */
  bool check_type(const Armor & armor) const;

  /** @brief 将原图缩放和填充为网络输入 @param raw_img 原始图像 @return 网络输入图像 */
  cv::Mat inference_image(const cv::Mat & raw_img);
  /** @brief 归一化装甲板中心坐标 @param bgr_img 输入图像 @param center 像素坐标 @return 归一化坐标 */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /** @brief 解析并执行 NMS @param scale 输入缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);
  /** @brief 将网络候选框转换为装甲板 @param color_id 颜色类别 @param class_id 数字类别 @param confidence 置信度 @param box 检测框 @param keypoints 关键点 @return 装甲板 */
  Armor make_armor(
    int color_id, int class_id, float confidence, const cv::Rect & box,
    const std::vector<cv::Point2f> & keypoints) const;

  /** @brief 返回数组最大元素下标 @param values 数组指针 @param count 元素数 @return 最大值下标 */
  static int argmax(const float * values, int count);
  /** @brief 计算 sigmoid @param value 输入值 @return sigmoid 结果 */
  static float sigmoid(float value) noexcept;

  /** @brief 保存装甲板样本 @param armor 装甲板 */
  void save(const Armor & armor) const;
  /** @brief 绘制检测结果 @param img 图像 @param armors 装甲板列表 @param frame_count 帧编号 */
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_HPP
