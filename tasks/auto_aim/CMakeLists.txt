cmake_minimum_required(VERSION 3.16)

set(AUTO_AIM_SOURCES
    armor.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    kf_example/ca_from_tju.cpp
    kf_example/rv_from_fyt.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    rv_detector.cpp
    rv_number.cpp
    planner/planner.cpp
    planner/planner_mpc.cpp
    planner/planner_trajectory.cpp
)

set(AUTO_AIM_LIBRARIES io)

if(OPENVINO_MAKE)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    list(APPEND AUTO_AIM_SOURCES
        classifier.cpp
        detector.cpp
        yolo.cpp
        ov_yolos/yolov5.cpp
        ov_yolos/yolov8.cpp
        ov_yolos/yolo11.cpp
        multithread/commandgener.cpp
        multithread/mt_detector.cpp
    )
    list(APPEND AUTO_AIM_LIBRARIES openvino::runtime)
    message(STATUS "--------------------OpenVINO_MAKE is ON--------------------")
elseif(TENSOR_RT_MAKE)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    list(APPEND AUTO_AIM_SOURCES
        yolo.cpp
        trt_yolos/trt_yolo_0708.cpp
        trt_yolos/trt_0708_kernel.cu
        trt_yolos/trt_yolo_0526.cpp
        trt_yolos/trt_0526_kernel.cu
    )
    list(APPEND AUTO_AIM_LIBRARIES
        ${OpenCV_LIBS}
        ${TENSORRT_INFER_LIBRARY}
        CUDA::cudart
        pthread
    )
    message(STATUS "--------------------TENSOR_RT_MAKE is ON--------------------")
else()
    message(STATUS "------------------OpenVINO_MAKE is OFF  TENSOR_RT_MAKE is OFF----------------------")
endif()

add_subdirectory(planner/tinympc)
list(APPEND AUTO_AIM_LIBRARIES tinympcstatic)

add_library(auto_aim OBJECT ${AUTO_AIM_SOURCES})
target_link_libraries(auto_aim PUBLIC ${AUTO_AIM_LIBRARIES})

if(OPENVINO_MAKE)
    target_compile_definitions(auto_aim PUBLIC OPENVINO_MAKE)
elseif(TENSOR_RT_MAKE)
    target_compile_definitions(auto_aim PUBLIC TENSOR_RT_MAKE)
    target_include_directories(auto_aim PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()
