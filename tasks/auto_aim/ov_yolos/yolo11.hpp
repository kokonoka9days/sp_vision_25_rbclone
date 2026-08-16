#ifndef AUTO_AIM__YOLO11_HPP
#define AUTO_AIM__YOLO11_HPP

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
class YOLO11 : public YOLOBase
{
public:
  /** @brief 初始化 OpenVINO YOLO11 检测器 @param config_path YAML 配置路径 @param debug 是否启用调试输出 */
  YOLO11(const std::string & config_path, bool debug);

  /** @brief 同步检测图像中的装甲板 @param bgr_img BGR 图像 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /** @brief 异步流水线检测帧 @param frame_data 帧数据 @param frame_count 帧编号 @return 最早完成的帧结果 */
  YOLOFrameData detect(YOLOFrameData frame_data, int frame_count) override;

  /** @brief 后处理网络输出 @param scale 输入缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  const int class_num_ = 38;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  OpenVINOAsyncPipeline async_pipeline_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  /** @brief 检查类别名称是否合法 @param armor 装甲板 @return 合法时返回 true */
  bool check_name(const Armor & armor) const;
  /** @brief 检查尺寸类型是否合法 @param armor 装甲板 @return 合法时返回 true */
  bool check_type(const Armor & armor) const;

  /** @brief 归一化装甲板中心坐标 @param bgr_img 输入图像 @param center 像素坐标 @return 归一化坐标 */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /** @brief 解析并执行 NMS @param scale 输入缩放系数 @param output 网络输出 @param bgr_img 原图 @param frame_count 帧编号 @return 装甲板列表 */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  /** @brief 保存装甲板样本 @param armor 装甲板 */
  void save(const Armor & armor) const;
  /** @brief 绘制检测结果 @param img 图像 @param armors 装甲板列表 @param frame_count 帧编号 */
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  /** @brief 将四个关键点排列为统一顺序 @param keypoints 待排序关键点 */
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLO11_HPP
