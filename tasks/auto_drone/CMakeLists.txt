cmake_minimum_required(VERSION 3.16)

# 1. 引入 tinympc 子目录，使得 cmake 去编译 tinympcstatic 库
add_subdirectory(tinympc)

add_library(auto_drone OBJECT 
    drone_armor.cpp
    drone_solver.cpp
    drone_target.cpp
    laser_ray_aim.cpp
    drone_planner.cpp
    drone_yolo.cpp     
    drone_tracker.cpp  
    cuda_preprocess.cu
)

target_include_directories(auto_drone SYSTEM PUBLIC ${TRT_INC})

target_link_libraries(auto_drone 
    io 
    tools
    ${TRT_LIB}
    cudart
    tinympcstatic      
)
