cmake_minimum_required(VERSION 3.16)

if(OPENVINO_MAKE)
    set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    add_library(auto_aim OBJECT 
        armor.cpp
        classifier.cpp 
        detector.cpp
        solver.cpp
        aimer.cpp
        target.cpp
        tracker.cpp
        voter.cpp
        shooter.cpp
        rv_detector.cpp
        rv_number.cpp
        yolo.cpp
        yolos/yolov5.cpp
        yolos/yolov8.cpp
        yolos/yolo11.cpp
        multithread/commandgener.cpp
        multithread/mt_detector.cpp
        planner/planner.cpp
    )
    add_subdirectory(planner/tinympc)
    target_link_libraries(auto_aim io openvino::runtime tinympcstatic)

    message(STATUS "--------------------OpenVINO_MAKE is ON--------------------")
else()
    add_library(auto_aim OBJECT 
        armor.cpp
        # classifier.cpp 
        # detector.cpp
        solver.cpp
        aimer.cpp
        target.cpp
        tracker.cpp
        voter.cpp
        shooter.cpp
        rv_detector.cpp
        rv_number.cpp
        # yolo.cpp
        # yolos/yolov5.cpp
        # yolos/yolov8.cpp
        # # yolos/yolo11.cpp
        # multithread/commandgener.cpp
        # multithread/mt_detector.cpp
        planner/planner.cpp
    )
    add_subdirectory(planner/tinympc)
    target_link_libraries(auto_aim io tinympcstatic)
    message(STATUS "--------------------OpenVINO_MAKE is OFF--------------------")
endif()
