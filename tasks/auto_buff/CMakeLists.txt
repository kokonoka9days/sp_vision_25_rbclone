cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    buff_aimer.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_type.cpp
    buff_target.cpp
    rm_buff_detector.cpp
    yolo11_buff.cpp
)

target_link_libraries(auto_buff auto_aim openvino::runtime ${CERES_LIBRARIES} VisCore_rune_detector)