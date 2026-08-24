#include <cmath>
#include <cstdlib>
#include <vector>

#include "rune_detector.hpp"

#define CHECK(condition) do { if (!(condition)) std::abort(); } while (false)

namespace
{
constexpr int channels = 18;
constexpr int anchors = 3;

void set_candidate(
  std::vector<float> & tensor, int anchor, int model_class, float score,
  float center_x, float center_y)
{
  tensor[model_class * anchors + anchor] = score;
  for (int point = 0; point < 5; ++point) {
    const int base = 3 + point * 3;
    const float x = center_x + static_cast<float>(point - 2) * 2.0f;
    const float y = center_y + static_cast<float>(point - 2) * 3.0f;
    tensor[base * anchors + anchor] = x * 0.5f + 10.0f;
    tensor[(base + 1) * anchors + anchor] = y * 0.5f + 20.0f;
    tensor[(base + 2) * anchors + anchor] = 0.9f;
  }
}

std::vector<float> transpose(const std::vector<float> & nca)
{
  std::vector<float> nac(nca.size());
  for (int channel = 0; channel < channels; ++channel) {
    for (int anchor = 0; anchor < anchors; ++anchor) {
      nac[anchor * channels + channel] = nca[channel * anchors + anchor];
    }
  }
  return nac;
}
}  // namespace

int main()
{
  std::vector<float> nca(channels * anchors, 0.0f);
  set_candidate(nca, 0, 0, 0.90f, 100.0f, 120.0f);
  set_candidate(nca, 1, 1, 0.80f, 102.0f, 121.0f);
  set_candidate(nca, 2, 2, 0.95f, 300.0f, 220.0f);

  const auto small = auto_buff::RuneDetector::decode_tensor(
    nca.data(), channels, anchors, true, 0.5f, 10, 20, cv::Size(640, 480),
    auto_buff::BuffMode::SMALL, 0.65f, 0.5f, 30.0f, 5);
  CHECK(small.size() == 1);
  CHECK(small.front().class_id == 0);
  CHECK(std::abs(small.front().keypoints[2].x - 100.0f) < 1e-5f);
  CHECK(std::abs(small.front().keypoints[2].y - 120.0f) < 1e-5f);

  const auto big = auto_buff::RuneDetector::decode_tensor(
    nca.data(), channels, anchors, true, 0.5f, 10, 20, cv::Size(640, 480),
    auto_buff::BuffMode::BIG, 0.65f, 0.5f, 30.0f, 5);
  CHECK(big.size() == 2);
  CHECK(big.front().class_id == 1);

  const std::vector<float> nac = transpose(nca);
  const auto transposed = auto_buff::RuneDetector::decode_tensor(
    nac.data(), anchors, channels, false, 0.5f, 10, 20, cv::Size(640, 480),
    auto_buff::BuffMode::SMALL, 0.65f, 0.5f, 30.0f, 5);
  CHECK(transposed.size() == small.size());
  CHECK(cv::norm(transposed.front().keypoints[2] - small.front().keypoints[2]) < 1e-5);
  return 0;
}
