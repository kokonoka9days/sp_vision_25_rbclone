cmake_minimum_required(VERSION 3.16.3)

project(sp_vision LANGUAGES CXX C CUDA)

# 强制 CUDA 使用 C++14 标准 (这是 nvcc 最喜欢的标准)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 确保 CUDA 真的使用 g++-10
set(CMAKE_CUDA_HOST_COMPILER g++-10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

# 生成 compile_commands.json，供 VS Code IntelliSense 使用 (消除红波浪线)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ================= 查找常规依赖 =================
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)

# ================= 查找 ROS 2 依赖 =================
# 这里改为 REQUIRED，如果没 source 环境变量，会在 CMake 阶段直接报错提醒
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
# ===============================================

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/io/serial/include)

include_directories("/opt/TensorRT-8.6.1.6/include")
link_directories("/opt/TensorRT-8.6.1.6/lib")

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_drone)

#################### src ###################
add_executable(rb_auto_drone_debug src/rb_auto_drone_debug.cpp)

# 链接常规依赖
target_link_libraries(rb_auto_drone_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_drone tools io nvinfer)

# 链接 ROS 2 依赖给主程序
ament_target_dependencies(rb_auto_drone_debug rclcpp std_msgs)

################## calibration ################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)
add_executable(test_simple calibration/test_simple.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(test_simple ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)