cmake_minimum_required(VERSION 3.16)

# 复用顶层已定义的 TRT_INC 和 TRT_LIB (TensorRT-10.11.0.33)
# 若顶层未定义则兜底查找
if(NOT TRT_INC OR NOT TRT_LIB)
    find_path(TRT_INC NvInfer.h
        PATHS /home/lzhros/TensorRT-10.11.0.33/include
              /usr/local/TensorRT-10.11.0.33/include
              /usr/include
    )
    find_library(TRT_LIB nvinfer
        PATHS /home/lzhros/TensorRT-10.11.0.33/lib
              /usr/local/TensorRT-10.11.0.33/lib
              /usr/lib/x86_64-linux-gnu
    )

    if(NOT TRT_INC OR NOT TRT_LIB)
        message(FATAL_ERROR "TensorRT not found! Please check your TensorRT installation.")
    endif()
endif()

# 1. 引入 tinympc 子目录，使得 cmake 去编译 tinympcstatic 库
add_subdirectory(tinympc)

add_library(auto_drone OBJECT 
    drone_armor.cpp
    drone_solver.cpp
    drone_target.cpp
    drone_planner.cpp
    drone_yolo.cpp     
    drone_tracker.cpp  
    cuda_preprocess.cu
)

# 2. 将 TensorRT 头文件目录包含进来
target_include_directories(auto_drone PUBLIC ${TRT_INC})

# 3. 链接 TensorRT 核心库 (nvinfer) 和 CUDA 运行时库 (cudart)，以及 tinympcstatic
target_link_libraries(auto_drone 
    io 
    tools
    ${TRT_LIB} 
    cudart
    tinympcstatic      
)
