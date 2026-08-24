#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "json.hpp"

#ifndef AUTO_BUFF_TEST_DEFAULT_CONFIG
#error "AUTO_BUFF_TEST_DEFAULT_CONFIG is required"
#endif

#define CHECK(condition) do { if (!(condition)) std::abort(); } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 3);
  PowerRuneConfigStore first;
  PowerRuneConfigStore second;
  first.initialize(AUTO_BUFF_TEST_DEFAULT_CONFIG, argv[1]);
  second.initialize(AUTO_BUFF_TEST_DEFAULT_CONFIG, argv[2]);

  CHECK(first.camera_matrix()(0, 0) > 0.0);
  CHECK(second.camera_matrix()(0, 0) > 0.0);
  CHECK(std::abs(first.R_camera2gimbal().determinant() - 1.0) < 0.05);
  CHECK(std::abs(second.R_camera2gimbal().determinant() - 1.0) < 0.05);
  CHECK(first.onnx_path().filename() == "model-0624.onnx");
  CHECK(first.engine_path().filename() == "model-0624-fp16.engine");
  CHECK(first.camera_matrix() != second.camera_matrix());
  return 0;
}
