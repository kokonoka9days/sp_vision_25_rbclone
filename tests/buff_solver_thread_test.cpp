#include <atomic>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

#include "tasks/auto_buff/buff_solver.hpp"

#define CHECK(condition)             \
  do {                               \
    if (!(condition)) std::abort();  \
  } while (false)

int main(int argc, char ** argv)
{
  CHECK(argc == 2);
  auto_buff::Solver solver(argv[1], auto_buff::BuffConfig{});
  std::atomic_bool valid{true};

  std::thread writer([&] {
    for (int i = 0; i < 1000; ++i) {
      const double angle = static_cast<double>(i) * 1e-4;
      solver.set_R_gimbal2world(
        Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ())));
    }
  });

  std::vector<std::thread> readers;
  for (int thread = 0; thread < 3; ++thread) {
    readers.emplace_back([&] {
      for (int i = 0; i < 1000; ++i) {
        const Eigen::Matrix3d rotation = solver.R_gimbal2world();
        if (!rotation.allFinite() || std::abs(rotation.determinant() - 1.0) > 1e-8) valid = false;
      }
    });
  }

  writer.join();
  for (auto & reader : readers) reader.join();
  CHECK(valid.load());
  return 0;
}
