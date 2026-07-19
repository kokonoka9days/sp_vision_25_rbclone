cmake_minimum_required(VERSION 3.16.3)

# 【修改 1】: 移除 CUDA 语言支持，只保留 C 和 CXX
project(sp_vision LANGUAGES CXX C)

# ==========================================
# 0. 核心编译选项配置 (参考雷达规范)
# ==========================================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)
endif()

# 强制全项目生成 -fPIC (对动态链接至关重要)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 【修改 2】: 删除了强制 CUDA 使用 C++14 和 g++-10 的限制，全项目统一使用 C++17
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

# 【修改 3】: 引入 OpenVINO 依赖
find_package(OpenVINO REQUIRED)

# ==========================================
# 2. 查找 ROS 2 依赖
# ==========================================
# 这里为 REQUIRED，如果没 source 环境变量，会在 CMake 阶段直接报错提醒
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# ==========================================
# 3. 头文件包含路径
# ==========================================
# 【修改 4】: 删除了原有的 TensorRT-10.11.0.33 搜索逻辑，并移除了 TRT_INC 和 CUDA 头文件挂载
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/io/serial/include
)

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_drone)

#################### src ###################
add_executable(rb_auto_drone_debug src/rb_auto_drone_debug.cpp)
add_executable(rb_auto_drone_diagnostics src/rb_auto_drone_diagnostics.cpp)
add_executable(rb_auto_drone_ros_debug src/rb_auto_drone_ros_debug.cpp)
add_executable(rb_auto_drone_video_test src/rb_auto_drone_video_test.cpp)

# 【修改 5】: 链接常规依赖，将底部的 ${TRT_LIB} 统一替换为 openvino::runtime
target_link_libraries(rb_auto_drone_debug 
    ${OpenCV_LIBS} 
    fmt::fmt 
    yaml-cpp 
    auto_drone 
    tools 
    io 
    openvino::runtime
)

target_link_libraries(rb_auto_drone_diagnostics
    ${OpenCV_LIBS}
    fmt::fmt
    yaml-cpp
    auto_drone
    tools
    io
    openvino::runtime
)

target_link_libraries(rb_auto_drone_ros_debug 
    ${OpenCV_LIBS} 
    fmt::fmt 
    yaml-cpp 
    auto_drone 
    tools 
    io 
    openvino::runtime
)

target_link_libraries(rb_auto_drone_video_test
    ${OpenCV_LIBS}
    fmt::fmt
    yaml-cpp
    auto_drone
    tools
    io
    openvino::runtime
)

# 链接 ROS 2 依赖给主程序 (标准做法)
ament_target_dependencies(rb_auto_drone_debug rclcpp std_msgs)
ament_target_dependencies(rb_auto_drone_ros_debug rclcpp std_msgs)

################## calibration ################
add_library(laser_ray_calibration STATIC calibration/laser_ray_calibration.cpp)
add_executable(calibrate_camera_laser_ray calibration/calibrate_camera_laser_ray.cpp)
add_executable(laser_ray_calibration_test calibration/laser_ray_calibration_test.cpp)
add_executable(
  drone_planner_laser_test
  tasks/auto_drone/laser_ray_aim_test.cpp
  tasks/auto_drone/laser_ray_aim.cpp)
add_executable(
  drone_planner_inline_config_test
  tasks/auto_drone/drone_planner_inline_config_test.cpp)
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)
add_executable(test_simple calibration/test_simple.cpp)
add_executable(run_calibration calibration/run_calibration.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(laser_ray_calibration PUBLIC ${OpenCV_LIBS} Ceres::ceres)
target_link_libraries(
  calibrate_camera_laser_ray
  ${OpenCV_LIBS} fmt::fmt yaml-cpp laser_ray_calibration tools io Ceres::ceres)
target_link_libraries(
  laser_ray_calibration_test ${OpenCV_LIBS} fmt::fmt laser_ray_calibration Ceres::ceres)
target_link_libraries(drone_planner_laser_test ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(
  drone_planner_inline_config_test
  ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_drone tools io openvino::runtime)
enable_testing()
add_test(NAME laser_ray_calibration_test COMMAND laser_ray_calibration_test)
add_test(NAME drone_planner_laser_test COMMAND drone_planner_laser_test)
add_test(NAME drone_planner_inline_config_test COMMAND drone_planner_inline_config_test)
# 【修改 6】: 校准工具同样将 ${TRT_LIB} 替换为 openvino::runtime
target_link_libraries(run_calibration ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_drone tools io openvino::runtime)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)
target_link_libraries(test_simple ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

# ==========================================
# 4. 注册 ROS 2 包
# ==========================================
ament_package()
