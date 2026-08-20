#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "yaw_delay_model.hpp"

namespace
{
constexpr double kDegreeToRad = 0.017453292519943295;

struct InputFile
{
  std::string path;
  enum class Kind { Positive, Negative, Reverse } kind;
};

std::vector<tools::YawDelaySample> longest_segment(const std::string & requested_path)
{
  std::filesystem::path path(requested_path);
  if (path.extension() == ".csv") path.replace_extension(".jsonl");
  std::ifstream input(path);
  if (!input.is_open()) throw std::runtime_error("cannot open " + path.string());

  std::vector<std::vector<tools::YawDelaySample>> segments(1);
  double previous_time = -1;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto record = nlohmann::json::parse(line);
    if (!record.contains("time_us") || !record.contains("data")) continue;
    const auto & data = record["data"];
    if (!data.contains("plan_yaw") || !data.contains("gimbal_yaw")) continue;

    const double time = static_cast<double>(record["time_us"].get<int64_t>()) / 1e6;
    const bool controlled = data.value("plan_mode", 1) != 0;
    if (!controlled || (previous_time >= 0 && time - previous_time > 0.03)) {
      if (!segments.back().empty()) segments.emplace_back();
      previous_time = time;
      if (!controlled) continue;
    }
    previous_time = time;
    segments.back().push_back({
      time, data["plan_yaw"].get<double>() * kDegreeToRad,
      data["gimbal_yaw"].get<double>() * kDegreeToRad});
  }

  const auto nonempty = std::find_if(
    segments.begin(), segments.end(), [](const auto & segment) { return !segment.empty(); });
  if (nonempty == segments.end()) return {};
  return *std::max_element(
    segments.begin(), segments.end(),
    [](const auto & lhs, const auto & rhs) { return lhs.size() < rhs.size(); });
}

const char * kind_name(InputFile::Kind kind)
{
  switch (kind) {
    case InputFile::Kind::Positive:
      return "positive";
    case InputFile::Kind::Negative:
      return "negative";
    case InputFile::Kind::Reverse:
      return "reverse";
  }
  return "unknown";
}

double estimate_sample_period(const std::vector<tools::YawDelaySample> & samples)
{
  std::vector<double> intervals;
  intervals.reserve(samples.size() > 1 ? samples.size() - 1 : 0);
  for (std::size_t i = 1; i < samples.size(); ++i) {
    const double interval = samples[i].time_s - samples[i - 1].time_s;
    if (std::isfinite(interval) && interval > 0) intervals.push_back(interval);
  }
  if (intervals.empty()) return 0.005;
  std::sort(intervals.begin(), intervals.end());
  return intervals[intervals.size() / 2];
}

void merge_duplicate_speeds(std::vector<std::pair<double, double>> & curve)
{
  if (curve.empty()) return;
  std::vector<std::pair<double, double>> merged;
  merged.reserve(curve.size());
  for (const auto point : curve) {
    if (!merged.empty() && point.first == merged.back().first) {
      merged.back().second = (merged.back().second + point.second) * 0.5;
    } else {
      merged.push_back(point);
    }
  }
  curve.swap(merged);
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: yaw_delay_analyzer [--positive|--negative|--reverse] record.jsonl [...]\n";
    return 1;
  }

  std::vector<InputFile> inputs;
  InputFile::Kind current_kind = InputFile::Kind::Positive;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--positive") {
      current_kind = InputFile::Kind::Positive;
    } else if (argument == "--negative") {
      current_kind = InputFile::Kind::Negative;
    } else if (argument == "--reverse") {
      current_kind = InputFile::Kind::Reverse;
    } else {
      inputs.push_back({argument, current_kind});
    }
  }
  if (inputs.empty()) return 1;

  std::vector<std::pair<double, double>> positive_curve;
  std::vector<std::pair<double, double>> negative_curve;
  std::vector<std::pair<double, double>> reverse_estimates;
  try {
    for (const auto & input : inputs) {
      const auto samples = longest_segment(input.path);
      const auto estimate = tools::estimate_yaw_delay(samples, estimate_sample_period(samples));
      if (!estimate.valid) {
        std::cerr << input.path << ": rejected: " << estimate.reason << '\n';
        continue;
      }
      std::cerr << input.path << " (" << kind_name(input.kind) << "): delay="
                << estimate.delay_s * 1000.0 << " ms, speed_p95=" << estimate.speed_rad_s
                << " rad/s, correlation=" << estimate.correlation << '\n';
      if (input.kind == InputFile::Kind::Positive) {
        positive_curve.emplace_back(estimate.speed_rad_s, estimate.delay_s);
      } else if (input.kind == InputFile::Kind::Negative) {
        negative_curve.emplace_back(estimate.speed_rad_s, estimate.delay_s);
      } else {
        reverse_estimates.emplace_back(estimate.speed_rad_s, estimate.delay_s);
      }
    }
  } catch (const std::exception & error) {
    std::cerr << "yaw_delay_analyzer: " << error.what() << '\n';
    return 1;
  }

  if (positive_curve.empty() || negative_curve.empty()) {
    std::cerr << "yaw_delay_analyzer: positive and negative curves both need valid data\n";
    return 2;
  }

  std::sort(positive_curve.begin(), positive_curve.end());
  std::sort(negative_curve.begin(), negative_curve.end());
  merge_duplicate_speeds(positive_curve);
  merge_duplicate_speeds(negative_curve);

  double reverse_penalty = 0;
  if (!reverse_estimates.empty()) {
    std::vector<double> penalties;
    for (const auto & [speed, delay] : reverse_estimates) {
      const auto nearest = [&](const auto & curve) {
        return *std::min_element(curve.begin(), curve.end(), [speed](const auto & lhs, const auto & rhs) {
          return std::abs(lhs.first - speed) < std::abs(rhs.first - speed);
        });
      };
      const auto positive = nearest(positive_curve);
      const auto negative = nearest(negative_curve);
      const double baseline =
        std::abs(positive.first - speed) < std::abs(negative.first - speed)
          ? positive.second
          : negative.second;
      penalties.push_back(std::max(0.0, delay - baseline));
    }
    std::sort(penalties.begin(), penalties.end());
    reverse_penalty = penalties[penalties.size() / 2];
    const auto max_curve_delay = [](const auto & curve) {
      return std::max_element(
        curve.begin(), curve.end(),
        [](const auto & lhs, const auto & rhs) { return lhs.second < rhs.second; })
        ->second;
    };
    reverse_penalty = std::min(
      reverse_penalty,
      std::max(
        0.0, 0.2 - std::max(max_curve_delay(positive_curve), max_curve_delay(negative_curve))));
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "yaw_delay_curve:\n  positive:\n";
  for (const auto & [speed, delay] : positive_curve) {
    std::cout << "    - [" << speed << ", " << delay << "]\n";
  }
  std::cout << "  negative:\n";
  for (const auto & [speed, delay] : negative_curve) {
    std::cout << "    - [" << speed << ", " << delay << "]\n";
  }
  std::cout << "yaw_reverse_penalty: " << reverse_penalty << "\n";
  return 0;
}
