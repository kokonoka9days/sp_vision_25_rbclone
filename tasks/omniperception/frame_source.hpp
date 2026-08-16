#ifndef OMNIPERCEPTION__FRAME_SOURCE_HPP
#define OMNIPERCEPTION__FRAME_SOURCE_HPP

#include <chrono>
#include <string>

#include <opencv2/core.hpp>

namespace omniperception
{
class IFrameSource
{
public:
  /** @brief 销毁帧源接口 */
  virtual ~IFrameSource() = default;
  /** @brief 在限定时间内读取一帧 @param image 输出图像 @param timestamp 输出时间戳 @param timeout 最长等待时间 @return 成功读取时返回 true */
  virtual bool read_for(
    cv::Mat & image, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) = 0;
  /** @brief 获取帧源名称 @return 名称只读引用 */
  virtual const std::string & name() const = 0;
};

}  // namespace omniperception

#endif  // OMNIPERCEPTION__FRAME_SOURCE_HPP
