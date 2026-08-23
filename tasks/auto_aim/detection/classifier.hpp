#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "../model/armor.hpp"

namespace auto_aim
{
class Classifier
{
public:
  /** @brief 根据配置加载数字分类模型 @param config_path YAML 配置文件路径 */
  explicit Classifier(const std::string & config_path);

  /** @brief 使用 OpenCV DNN 对装甲板数字分类 @param armor 待分类装甲板，结果写回 name 和 confidence */
  void classify(Armor & armor);

  /** @brief 使用 OpenVINO 对装甲板数字分类 @param armor 待分类装甲板，结果写回 name 和 confidence */
  void ovclassify(Armor & armor);

private:
  cv::dnn::Net net_;
  ov::Core core_;
  ov::CompiledModel compiled_model_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP
