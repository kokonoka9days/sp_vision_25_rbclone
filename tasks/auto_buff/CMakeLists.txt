cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    active_fans_detector.cpp
    buff_type.cpp
    buff_track_bank.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    yolo11_buff.cpp
    kami_rune/fan_skeleton_extractor.cpp
)

target_link_libraries(auto_buff auto_aim openvino::runtime ${CERES_LIBRARIES})
