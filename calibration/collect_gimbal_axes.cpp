#include <fmt/core.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <string>
#include <vector>

#include "io/gimbal/gimbal.hpp"
#include "tools/math_tools.hpp"

namespace
{

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr auto kSampleDuration = std::chrono::milliseconds(800);

struct PoseSample
{
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  double maximum_deviation_deg = 0.0;
  int count = 0;
};

void wait_for_enter(const std::string & prompt)
{
  fmt::print("\n{}\n准备好后按回车开始自动采样（采样期间不要碰云台）...", prompt);
  std::fflush(stdout);
  std::string line;
  std::getline(std::cin, line);
}

double quaternion_distance_deg(const Eigen::Quaterniond & lhs, const Eigen::Quaterniond & rhs)
{
  const double cosine = std::clamp(std::abs(lhs.normalized().dot(rhs.normalized())), 0.0, 1.0);
  return 2.0 * std::acos(cosine) * kRadToDeg;
}

PoseSample sample_pose(io::Gimbal & gimbal)
{
  std::vector<Eigen::Quaterniond> samples;
  const auto end = std::chrono::steady_clock::now() + kSampleDuration;
  while (std::chrono::steady_clock::now() < end || samples.size() < 10) {
    samples.push_back(gimbal.q(std::chrono::steady_clock::now()).normalized());
  }

  Eigen::Matrix4d accumulator = Eigen::Matrix4d::Zero();
  for (const auto & q : samples) {
    const Eigen::Vector4d coefficients = q.coeffs();  // xyzw
    accumulator += coefficients * coefficients.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(accumulator);
  Eigen::Quaterniond average;
  average.coeffs() = solver.eigenvectors().col(3);
  average.normalize();

  double maximum_deviation_deg = 0.0;
  for (const auto & q : samples) {
    maximum_deviation_deg =
      std::max(maximum_deviation_deg, quaternion_distance_deg(average, q));
  }
  return {average, maximum_deviation_deg, static_cast<int>(samples.size())};
}

void print_pose(const std::string & name, const PoseSample & sample)
{
  const Eigen::Vector3d zyx_deg = tools::eulers(sample.q, 2, 1, 0) * kRadToDeg;
  fmt::print(
    "{}：Z={:+.3f} deg, Y={:+.3f} deg, X={:+.3f} deg，样本={}，最大抖动={:.4f} deg\n",
    name, zyx_deg.x(), zyx_deg.y(), zyx_deg.z(), sample.count, sample.maximum_deviation_deg);
  if (sample.maximum_deviation_deg > 0.15) {
    fmt::print("  [警告] 该姿态采样时云台仍在运动或抖动，建议重新运行并等待稳定。\n");
  }
}

struct AxisMeasurement
{
  double angle_deg = 0.0;
  Eigen::Vector3d axis_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis_in_reference_imu = Eigen::Vector3d::Zero();
};

AxisMeasurement measure_axis(
  const Eigen::Quaterniond & reference, const Eigen::Quaterniond & moved)
{
  const Eigen::Quaterniond relative = (moved * reference.conjugate()).normalized();
  const Eigen::AngleAxisd angle_axis(relative);
  const Eigen::Vector3d axis_world = angle_axis.axis().normalized();
  return {
    angle_axis.angle() * kRadToDeg, axis_world,
    (reference.conjugate() * axis_world).normalized()};
}

void print_vector(const std::string & key, const Eigen::Vector3d & value)
{
  fmt::print("{}: [{:.9f}, {:.9f}, {:.9f}]\n", key, value.x(), value.y(), value.z());
}

}  // namespace

const std::string keys =
  "{help h usage ? |                         | 输出命令行帮助}"
  "{@config-path   | configs/auto_drone.yaml | YAML 配置文件路径}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const std::string config_path = cli.get<std::string>(0);

  fmt::print(
    "云台真实旋转轴采集程序\n"
    "- 请关闭自动瞄准和激光，只保留手动云台控制。\n"
    "- IMU 必须随相机一起运动。\n"
    "- 每次只动提示指定的一个电机，另一个电机保持不动。\n");

  io::Gimbal gimbal(config_path);

  wait_for_enter("第1步：把两个电机都放在正常回中位置，等待云台完全静止。");
  const PoseSample neutral = sample_pose(gimbal);
  print_pose("初始回中", neutral);

  wait_for_enter(
    "第2步：只转动【上方电机】约 5～10 度；左侧电机不要动，然后等待完全静止。");
  const PoseSample top_moved = sample_pose(gimbal);
  print_pose("上方电机移动后", top_moved);

  wait_for_enter("第3步：把上方电机恢复到第1步的回中位置，两个电机都不要动。");
  const PoseSample returned = sample_pose(gimbal);
  print_pose("再次回中", returned);

  wait_for_enter(
    "第4步：只转动【左侧电机】约 5～10 度；上方电机不要动，然后等待完全静止。");
  const PoseSample left_moved = sample_pose(gimbal);
  print_pose("左侧电机移动后", left_moved);

  const AxisMeasurement top_axis = measure_axis(neutral.q, top_moved.q);
  const AxisMeasurement left_axis = measure_axis(returned.q, left_moved.q);
  const double return_error_deg = quaternion_distance_deg(neutral.q, returned.q);
  const double axes_angle_deg =
    std::acos(std::clamp(std::abs(top_axis.axis_in_reference_imu.dot(
                           left_axis.axis_in_reference_imu)),
                        0.0, 1.0)) *
    kRadToDeg;

  fmt::print("\n================ 复制下面全部结果给 Codex ================\n");
  fmt::print("top_motor_delta_deg: {:.6f}\n", top_axis.angle_deg);
  print_vector("top_axis_in_neutral_imu", top_axis.axis_in_reference_imu);
  print_vector("top_axis_in_world", top_axis.axis_in_world);
  fmt::print("left_motor_delta_deg: {:.6f}\n", left_axis.angle_deg);
  print_vector("left_axis_in_neutral_imu", left_axis.axis_in_reference_imu);
  print_vector("left_axis_in_world", left_axis.axis_in_world);
  fmt::print("neutral_return_error_deg: {:.6f}\n", return_error_deg);
  fmt::print("motor_axes_included_angle_deg: {:.6f}\n", axes_angle_deg);
  fmt::print("===========================================================\n");

  bool valid = true;
  if (top_axis.angle_deg < 2.0 || left_axis.angle_deg < 2.0) {
    fmt::print(stderr, "[失败] 电机转动量不足2度，请按提示转动5～10度后重新采集。\n");
    valid = false;
  }
  if (top_axis.angle_deg > 30.0 || left_axis.angle_deg > 30.0) {
    fmt::print(stderr, "[失败] 电机转动量超过30度，请减小到5～10度后重新采集。\n");
    valid = false;
  }
  if (return_error_deg > 1.0) {
    fmt::print(stderr, "[警告] 第3步没有准确回中；建议重新运行以提高旋转轴精度。\n");
  }
  if (std::abs(axes_angle_deg - 90.0) > 5.0) {
    fmt::print(stderr, "[警告] 两个电机轴夹角明显偏离90度，请检查是否每次只动了一个电机。\n");
  }
  return valid ? 0 : 2;
}
