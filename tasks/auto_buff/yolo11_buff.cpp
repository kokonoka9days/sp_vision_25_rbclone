#include "yolo11_buff.hpp"
#include <openvino/core/preprocess/pre_post_process.hpp> // 新增 PPP 头文件

const double ConfidenceThreshold = 0.7f;
const double IouThreshold = 0.4f;
namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path = yaml["model"].as<std::string>();
  
  // 1. 读取模型
  model = core.read_model(model_path);

  // ======================= 新增的 PPP 预处理代码 开始 =======================
  ov::preprocess::PrePostProcessor ppp(model);

  ppp.input()
      .tensor()
      .set_element_type(ov::element::u8)  // OpenCV 的 Mat 默认是 uint8
      .set_layout("NHWC")                 // OpenCV 默认是 NHWC 格式
      .set_color_format(ov::preprocess::ColorFormat::BGR); 

  ppp.input()
      .preprocess()
      .convert_element_type(ov::element::f32) 
      .convert_color(ov::preprocess::ColorFormat::RGB) 
      .scale({255.0, 255.0, 255.0});      

  ppp.input().model().set_layout("NCHW");

  model = ppp.build();
  // ======================= 新增的 PPP 预处理代码 结束 =======================

  /// 2. 载入并编译模型 (替换为你需要的 LATENCY 低延迟模式)
  compiled_model = core.compile_model(model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  
  /// 3. 创建推理请求
  infer_request = compiled_model.create_infer_request();
  
  // 4. 获取模型输入节点
  input_tensor = infer_request.get_input_tensor();
  
  // 自动从预处理后的模型中获取输入形状，再也不用手动改数字了！
auto input_shape = compiled_model.input().get_shape();
// 因为我们加了 PPP 把输入设为 NHWC，所以这里的 input_shape 通常就是 {1, H, W, 3}
input_tensor.set_shape(input_shape);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();  // 设置模型输入

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<YOLO11_BUFF::Object> ();
  }

  /// 预处理：调用统一的图像填充转换函数，并获取正确的缩放 factor
  const float factor = fill_tensor_data_image(input_tensor, image);

  /// 执行推理计算
  infer_request.infer();

  /// 处理推理计算结果
  const ov::Tensor output = infer_request.get_output_tensor();  // 获得推理结果
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output_shape[1];  // 获得"output"节点的rows 15
  const int out_cols = output_shape[2];  // 获得"output"节点的cols 8400
  const cv::Mat det_output(
    out_rows, out_cols, CV_32F, (float *)output_buffer);  // output_buff类型转换
  std::vector<cv::Rect> boxes;                            // 目标框
  std::vector<float> confidences;                         // 置信度
  std::vector<std::vector<float>> objects_keypoints;      // 关键点
  // 输出格式是[15,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是[cx, cy, ow, oh], 中间score, 最后5*2关键点(3代表每个关键点的信息, 包括[x, y, visibility],如果是2，则没有visibility)
  // 15 = 4 + 1 + NUM_POINTS * 2      56
  for (int i = 0; i < det_output.cols; ++i) {
    const float score = det_output.at<float>(4, i);
    // 如果置信度满足条件则放进vector
    if (score > ConfidenceThreshold) {
      // 获取目标框
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      cv::Rect box;
      box.x = static_cast<int>((cx - 0.5 * ow) * factor);
      box.y = static_cast<int>((cy - 0.5 * oh) * factor);
      box.width = static_cast<int>(ow * factor);
      box.height = static_cast<int>(oh * factor);
      boxes.push_back(box);

      // 获取置信度
      confidences.push_back(score);

      // 获取关键点
     std::vector<float> keypoints;
      cv::Mat kpts = det_output.col(i).rowRange(5, 5 + NUM_POINTS * 2);
      for (int j = 0; j < NUM_POINTS; ++j) {
        const float x = kpts.at<float>(j * 2 + 0, 0) * factor;
        const float y = kpts.at<float>(j * 2 + 1, 0) * factor;
        // const float s = kpts.at<float>(j * 3 + 2, 0);
        keypoints.push_back(x);
        keypoints.push_back(y);
        // keypoints.push_back(s);
      }
      objects_keypoints.push_back(keypoints);
    }
  }

  /// NMS,消除具有较低置信度的冗余重叠框,用于处理多个框的情况
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);

  std::vector<Object> object_result;  // 最终得到的object
  for (size_t i = 0; i < indexes.size(); ++i) {
    Object obj;
    const int index = indexes[i];
    obj.rect = boxes[index];
    obj.prob = confidences[index];

    const std::vector<float> & keypoint = objects_keypoints[index];
    for (int j = 0; j < NUM_POINTS; ++j) {
      const float x_coord = keypoint[j * 2];
      const float y_coord = keypoint[j * 2 + 1];
      obj.kpt.push_back(cv::Point2f(x_coord, y_coord));
    }
    object_result.push_back(obj);

    /// 绘制关键点和连线
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);            // 绘制矩形框
    const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);  // 绘制标签
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
      obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    const int radius = 2;  // 绘制关键点
    const cv::Size & shape = image.size();
    for (int k = 0; k < NUM_POINTS; ++k)
      cv::circle(image, obj.kpt[k], radius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  }
  /// 计算FPS
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);

  // #ifdef SAVE
  //         save("save", image);
  // #endif
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();  // 设置模型输入

  /// 预处理
  const float factor = fill_tensor_data_image(input_tensor, image);  // 填充图片到合适的input size

  /// 执行推理计算

  infer_request.infer();

  /// 处理推理计算结果  output 输出格式是[17,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是[cx, cy, ow, oh], 中间score, 最后6*2关键点

  const ov::Tensor output = infer_request.get_output_tensor();  // 获得推理结果
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output_shape[1];  // 获得"output"节点的rows 17
  const int out_cols = output_shape[2];  // 获得"output"节点的cols 8400
  const cv::Mat det_output(
    out_rows, out_cols, CV_32F, (float *)output_buffer);  // output_buff类型转换

  /// 寻找置信度最大的框

  int best_index = -1;
  float max_confidence = 0.0f;
  for (int i = 0; i < det_output.cols; ++i) {
    const float confidence = det_output.at<float>(4, i);
    if (confidence > max_confidence) {
      max_confidence = confidence;
      best_index = i;
    }
  }
  std::vector<Object> object_result;  // 最终得到的object
  if (max_confidence > ConfidenceThreshold) {
    Object obj;
    // 获取目标框
    const float cx = det_output.at<float>(0, best_index);
    const float cy = det_output.at<float>(1, best_index);
    const float ow = det_output.at<float>(2, best_index);
    const float oh = det_output.at<float>(3, best_index);
    obj.rect.x = static_cast<int>((cx - 0.5 * ow) * factor);
    obj.rect.y = static_cast<int>((cy - 0.5 * oh) * factor);
    obj.rect.width = static_cast<int>(ow * factor);
    obj.rect.height = static_cast<int>(oh * factor);
    // 获取置信度
    obj.prob = max_confidence;
    // 获取关键点
    cv::Mat kpts = det_output.col(best_index).rowRange(5, 5 + NUM_POINTS * 2);
    for (int i = 0; i < NUM_POINTS; ++i) {
      const float x = kpts.at<float>(i * 2 + 0, 0) * factor;
      const float y = kpts.at<float>(i * 2 + 1, 0) * factor;
      obj.kpt.push_back(cv::Point2f(x, y));
    }
    object_result.push_back(obj);

    /// 0.3-0.7 save(这是用来继续训练数据集用的)
    // if (max_confidence < 0.7) save(std::to_string(start), image);

    /// 绘制关键点和连线
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);                  // 绘制矩形框
    const std::string label = "buff:" + std::to_string(max_confidence).substr(0, 4);  // 绘制标签
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
      obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    const int radius = 2;  // 绘制关键点
    const cv::Size & shape = image.size();
    for (int i = 0; i < NUM_POINTS; ++i) {
      cv::circle(image, obj.kpt[i], radius, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      cv::putText(
        image, std::to_string(i + 1), obj.kpt[i] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  /// 计算FPS
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;  // 归一化到[0, 1]
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

float YOLO11_BUFF::fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  // 因为使用了 PPP，tensor_shape 现在是 NHWC 格式，即 [1, 640, 640, 3]
  const size_t height = tensor_shape[1];
  const size_t width = tensor_shape[2];

  // 计算缩放因子 (letterbox)
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{
    scale, 0.0, 0.0, 0.0, scale, 0.0,
  };
  
  cv::Mat blob_image;
  // 直接进行仿射变换缩放
  cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));

  // 【提速核心】：因为预处理已经交给了 OpenVINO，
  // 这里直接把 uint8_t 的 BGR 像素数据通过 memcpy 内存拷贝进 tensor 即可。
  // 彻底去掉了原来极其耗时的 3 层 for 循环！
  uint8_t* input_tensor_data = input_tensor.data<uint8_t>();
  std::memcpy(input_tensor_data, blob_image.data, blob_image.total() * blob_image.elemSize());

  return 1 / scale;
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;

    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;

    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;

    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;

    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;

    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;

    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff