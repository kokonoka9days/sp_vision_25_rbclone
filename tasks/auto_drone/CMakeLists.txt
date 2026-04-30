cmake_minimum_required(VERSION 3.16)

# 查找 TensorRT 的头文件 (NvInfer.h)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS 
        /opt/TensorRT-8.6.1.6/include    # 您的 TensorRT 头文件实际位置 (优先搜索)
        /usr/include/x86_64-linux-gnu 
        /usr/include                     # 系统 CUDA 头文件常见位置
        /usr/local/cuda/include
)

# 查找 TensorRT 的核心库 (libnvinfer.so)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS 
        /opt/TensorRT-8.6.1.6/lib        # 您的 TensorRT 库文件实际位置 (优先搜索)
        /usr/lib/x86_64-linux-gnu 
        /usr/local/cuda/lib64
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found! Please check your TensorRT installation.")
endif()

# 其他已有配置...
# set(OpenVINO_DIR ...)
# find_package(OpenVINO REQUIRED ...)

# 1. 引入 tinympc 子目录，使得 cmake 去编译 tinympcstatic 库
add_subdirectory(tinympc)

add_library(auto_drone OBJECT 
    drone_armor.cpp
    drone_solver.cpp
    drone_target.cpp
    drone_planner.cpp
    drone_yolo.cpp     # 确保加入了 yolo 源文件
    drone_tracker.cpp  # <--- 加上这一行，解决 Tracker 报错
    cuda_preprocess.cu
)

# 2. 将 TensorRT 头文件目录包含进来
target_include_directories(auto_drone PUBLIC ${TENSORRT_INCLUDE_DIR})

# 3. 链接 TensorRT 核心库 (nvinfer) 和 CUDA 运行时库 (cudart)，以及刚刚引入的 tinympcstatic
target_link_libraries(auto_drone 
    io 
    tools
    ${TENSORRT_LIBRARY} 
    cudart
    tinympcstatic      # <--- 新增这行，链接 MPC 求解器静态库
)