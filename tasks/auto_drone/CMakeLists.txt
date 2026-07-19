cmake_minimum_required(VERSION 3.16)

# 寻找 OpenVINO 库 (替换了原有的 TensorRT 查找逻辑)
find_package(OpenVINO REQUIRED)

# 1. 引入 tinympc 子目录，使得 cmake 去编译 tinympcstatic 库
add_subdirectory(tinympc)

# 移除了 cuda_preprocess.cu
add_library(auto_drone OBJECT 
    drone_armor.cpp
    drone_solver.cpp
    drone_target.cpp
    laser_ray_aim.cpp
    drone_planner.cpp
    drone_yolo.cpp     
    drone_tracker.cpp  
)

# 2. 链接库
# 移除了 TRT_LIB, cudart; 换成 openvino::runtime
target_link_libraries(auto_drone 
    io 
    tools
    openvino::runtime 
    tinympcstatic      
)
