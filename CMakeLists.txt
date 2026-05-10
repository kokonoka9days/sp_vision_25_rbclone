cmake_minimum_required(VERSION 3.16.3)

project(sp_vision LANGUAGES CXX C CUDA)

# ==========================================
# 0. 核心编译选项配置 (参考雷达规范)
# ==========================================
# 忽略 TensorRT 10.x 的弃用警告，保持终端干净
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)
endif()

# 强制全项目生成 -fPIC (对动态链接至关重要)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# ==========================================
# 1. 查找常规依赖
# ==========================================
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)

# ==========================================
# 2. 查找 ROS 2 依赖
# ==========================================
# 这里为 REQUIRED，如果没 source 环境变量，会在 CMake 阶段直接报错提醒
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# ==========================================
# 3. 外部依赖系统路径配置 (同步雷达项目的 TensorRT-10.11.0.33)
# ==========================================
# 抛弃 8.6.1.6，使用雷达项目中的 10.11.0.33 绝对路径进行安全挂载
find_path(TRT_INC NvInfer.h PATHS /home/lzhros/TensorRT-10.11.0.33/include /usr/local/TensorRT-10.11.0.33/include /usr/include)
find_library(TRT_LIB nvinfer PATHS /home/lzhros/TensorRT-10.11.0.33/lib /usr/local/TensorRT-10.11.0.33/lib /usr/lib/x86_64-linux-gnu)

if(TRT_INC AND TRT_LIB)
    message(STATUS "Found TensorRT 10.11 include: ${TRT_INC}")
    message(STATUS "Found TensorRT 10.11 library: ${TRT_LIB}")
else()
    message(FATAL_ERROR "TensorRT 10.11 not found! 请检查 /home/lzhros/TensorRT-10.11.0.33 路径是否正确。")
endif()

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/io/serial/include
    ${TRT_INC}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_drone)

#################### src ###################
add_executable(rb_auto_drone_debug src/rb_auto_drone_debug.cpp)
add_executable(rb_auto_drone_ros_debug src/rb_auto_drone_ros_debug.cpp)

# 链接常规依赖 (使用安全找到的 10.11 版 ${TRT_LIB})
target_link_libraries(rb_auto_drone_debug 
    ${OpenCV_LIBS} 
    fmt::fmt 
    yaml-cpp 
    auto_drone 
    tools 
    io 
    ${TRT_LIB}
)

target_link_libraries(rb_auto_drone_ros_debug 
    ${OpenCV_LIBS} 
    fmt::fmt 
    yaml-cpp 
    auto_drone 
    tools 
    io 
    ${TRT_LIB}
)

# 链接 ROS 2 依赖给主程序 (标准做法)
ament_target_dependencies(rb_auto_drone_debug rclcpp std_msgs)
ament_target_dependencies(rb_auto_drone_ros_debug rclcpp std_msgs)

################## calibration ################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)
add_executable(test_simple calibration/test_simple.cpp)
add_executable(run_calibration calibration/run_calibration.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(run_calibration ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_drone tools io ${TRT_LIB})
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(test_simple ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

# ==========================================
# 4. 注册 ROS 2 包
# ==========================================
ament_package()