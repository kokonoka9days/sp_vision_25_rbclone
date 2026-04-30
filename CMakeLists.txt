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

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/io/serial/include)

include_directories("/opt/TensorRT-8.6.1.6/include")
link_directories("/opt/TensorRT-8.6.1.6/lib")

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_drone)

####################src###################
add_executable(rb_auto_drone_debug  src/rb_auto_drone_debug.cpp)

target_link_libraries(rb_auto_drone_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_drone tools io nvinfer)

##################calibration################
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

##################tests##################
# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

# 只有在找到 ROS 环境时才会添加
if(ament_cmake_FOUND)
    message(STATUS "找到ament_cmake_FOUND")
endif()
if(rclcpp_FOUND)
    message(STATUS "找到rclcpp_FOUND")
endif()
if(rosidl_typesupport_cpp_FOUND)
    message(STATUS "找到rosidl_typesupport_cpp_FOUND")
endif()
if(sp_msgs_FOUND)
    message(STATUS "找到sp_msgs_FOUND")
endif()

if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 当前项目中 src 和 tests 的相关 ROS 源码均不存在，相关构建项已移除
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()