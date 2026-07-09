#include "yolo11_buff.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace auto_buff
{
namespace
{
cv::Rect clip_rect(const cv::Rect & rect, const cv::Size & size)
{
  return rect & cv::Rect(0, 0, size.width, size.height);
}

cv::Point2f rect_center(const cv::Rect_<float> & rect)
{
  return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
}

cv::Scalar color_for_label(int label)
{
  static const std::array<cv::Scalar, 3> colors = {
    cv::Scalar(0, 220, 255), cv::Scalar(255, 160, 0), cv::Scalar(80, 255, 80)};
  if (label < 0 || label >= static_cast<int>(colors.size())) return {255, 255, 255};
  return colors[label];
}
}  // namespace

YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path = yaml["model"].as<std::string>();
  if (yaml["buff_confidence_threshold"]) {
    confidence_threshold_ = yaml["buff_confidence_threshold"].as<float>();
  }
  if (yaml["buff_keypoint_threshold"]) {
    keypoint_threshold_ = yaml["buff_keypoint_threshold"].as<float>();
  }
  if (yaml["buff_iou_threshold"]) iou_threshold_ = yaml["buff_iou_threshold"].as<float>();

  model = core.read_model(model_path);

  ov::preprocess::PrePostProcessor ppp(model);
  ppp.input()
    .tensor()
    .set_element_type(ov::element::u8)
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  ppp.input()
    .preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale({255.0, 255.0, 255.0});
  ppp.input().model().set_layout("NCHW");
  model = ppp.build();

  compiled_model =
    core.compile_model(model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  infer_request = compiled_model.create_infer_request();
  input_tensor = infer_request.get_input_tensor();
  input_tensor.set_shape(compiled_model.input().get_shape());
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  const float factor = fill_tensor_data_image(input_tensor, image);
  infer_request.infer();

  const ov::Tensor output = infer_request.get_output_tensor();
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = static_cast<int>(output_shape[1]);
  const int out_cols = static_cast<int>(output_shape[2]);
  const int expect_rows = 4 + NUM_CLASSES + NUM_POINTS * KPT_DIMS;
  if (out_rows < expect_rows) {
    tools::logger()->warn(
      "[YOLO11_BUFF] Unexpected output rows: {}, expected at least {}", out_rows, expect_rows);
    return {};
  }

  const cv::Mat det_output(out_rows, out_cols, CV_32F, const_cast<float *>(output_buffer));
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> labels;
  std::vector<std::vector<cv::Point2f>> keypoints;
  std::vector<std::vector<float>> keypoint_confidences;

  for (int i = 0; i < det_output.cols; ++i) {
    int label = -1;
    float score = -std::numeric_limits<float>::max();
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
      const float cls_score = det_output.at<float>(4 + cls, i);
      if (cls_score > score) {
        score = cls_score;
        label = cls;
      }
    }
    if (score < confidence_threshold_) continue;

    const float cx = det_output.at<float>(0, i);
    const float cy = det_output.at<float>(1, i);
    const float ow = det_output.at<float>(2, i);
    const float oh = det_output.at<float>(3, i);
    cv::Rect box;
    box.x = static_cast<int>((cx - 0.5f * ow) * factor);
    box.y = static_cast<int>((cy - 0.5f * oh) * factor);
    box.width = static_cast<int>(ow * factor);
    box.height = static_cast<int>(oh * factor);
    box = clip_rect(box, image.size());
    if (box.empty()) continue;

    std::vector<cv::Point2f> kpts;
    std::vector<float> kpts_conf;
    const int kpt_offset = 4 + NUM_CLASSES;
    for (int j = 0; j < NUM_POINTS; ++j) {
      const float x = det_output.at<float>(kpt_offset + j * KPT_DIMS + 0, i) * factor;
      const float y = det_output.at<float>(kpt_offset + j * KPT_DIMS + 1, i) * factor;
      const float conf = det_output.at<float>(kpt_offset + j * KPT_DIMS + 2, i);
      kpts.emplace_back(x, y);
      kpts_conf.emplace_back(conf);
    }

    boxes.push_back(box);
    confidences.push_back(score);
    labels.push_back(label);
    keypoints.push_back(kpts);
    keypoint_confidences.push_back(kpts_conf);
  }

  std::vector<int> kept_indexes;
  for (int cls = 0; cls < NUM_CLASSES; ++cls) {
    std::vector<cv::Rect> cls_boxes;
    std::vector<float> cls_confidences;
    std::vector<int> cls_indexes;
    for (size_t i = 0; i < labels.size(); ++i) {
      if (labels[i] != cls) continue;
      cls_boxes.push_back(boxes[i]);
      cls_confidences.push_back(confidences[i]);
      cls_indexes.push_back(static_cast<int>(i));
    }

    std::vector<int> nms_indexes;
    cv::dnn::NMSBoxes(
      cls_boxes, cls_confidences, confidence_threshold_, iou_threshold_, nms_indexes);
    for (int index : nms_indexes) kept_indexes.push_back(cls_indexes[index]);
  }

  std::sort(kept_indexes.begin(), kept_indexes.end(), [&](int lhs, int rhs) {
    return confidences[lhs] > confidences[rhs];
  });

  std::vector<Object> object_result;
  for (int index : kept_indexes) {
    Object obj;
    obj.rect = boxes[index];
    obj.label = labels[index];
    obj.prob = confidences[index];
    obj.kpt = keypoints[index];
    obj.kpt_conf = keypoint_confidences[index];
    object_result.push_back(obj);

    const auto color = color_for_label(obj.label);
    cv::rectangle(image, obj.rect, color, 1, 8);
    std::string label = class_names[obj.label] + ":" + std::to_string(obj.prob).substr(0, 4);
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, nullptr);
    const cv::Point text_org(
      static_cast<int>(obj.rect.x), std::max(12, static_cast<int>(obj.rect.y) - 4));
    cv::rectangle(
      image,
      cv::Rect(text_org.x, text_org.y - text_size.height, text_size.width, text_size.height + 4),
      color, cv::FILLED);
    cv::putText(
      image, label, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);

    for (int k = 0; k < NUM_POINTS; ++k) {
      const auto kpt_color =
        obj.kpt_conf[k] >= keypoint_threshold_ ? color : cv::Scalar(80, 80, 80);
      cv::circle(image, obj.kpt[k], 2, kpt_color, -1, cv::LINE_AA);
      cv::putText(
        image, std::to_string(k), obj.kpt[k] + cv::Point2f(4, -4), cv::FONT_HERSHEY_SIMPLEX,
        0.35, kpt_color, 1, cv::LINE_AA);
    }
    cv::circle(image, rect_center(obj.rect), 2, color, -1, cv::LINE_AA);
  }

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0f / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  auto objects = get_multicandidateboxes(image);
  if (objects.empty()) return {};

  auto best = std::max_element(objects.begin(), objects.end(), [](const Object & a, const Object & b) {
    return a.prob < b.prob;
  });
  return {*best};
}

void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

float YOLO11_BUFF::fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t height = tensor_shape[1];
  const size_t width = tensor_shape[2];

  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{scale, 0.0f, 0.0f, 0.0f, scale, 0.0f};

  cv::Mat blob_image;
  cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));

  uint8_t * input_tensor_data = input_tensor.data<uint8_t>();
  std::memcpy(input_tensor_data, blob_image.data, blob_image.total() * blob_image.elemSize());
  return 1.0f / scale;
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;
    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;
    std::cout << "        input type: " << input.get_element_type() << std::endl;
    std::cout << "        input shape: " << input.get_shape() << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;
    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;
    std::cout << "        output type: " << output.get_element_type() << std::endl;
    std::cout << "        output shape: " << output.get_shape() << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) std::filesystem::create_directories(saveDir);
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff
