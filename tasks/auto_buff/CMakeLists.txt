cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)

file(GLOB_RECURSE RP_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/rp_core/src/*.cpp)

set(AUTO_BUFF_SOURCES
    rune_system.cpp
    adapter/pose_adapter.cpp
    detector/rune_detector.cpp
    ${RP_CORE_SOURCES})

if(TENSOR_RT_MAKE)
    enable_language(CUDA)
    list(APPEND AUTO_BUFF_SOURCES detector/trt_rune_detector_kernel.cu)
endif()

add_library(auto_buff STATIC ${AUTO_BUFF_SOURCES})
target_compile_features(auto_buff PUBLIC cxx_std_17)
target_compile_definitions(auto_buff PRIVATE
    AUTO_BUFF_DEFAULT_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/power_rune_defaults.yaml")
target_include_directories(auto_buff
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/detector
      ${CMAKE_CURRENT_SOURCE_DIR}/rp_core/include
      ${CMAKE_CURRENT_SOURCE_DIR}/rp_core/config)
target_link_libraries(auto_buff
    PUBLIC Eigen3::Eigen
    PRIVATE ${OpenCV_LIBS} yaml-cpp Ceres::ceres tools)

if(OPENVINO_MAKE)
    target_compile_definitions(auto_buff PRIVATE SP_AUTO_BUFF_OPENVINO)
    target_link_libraries(auto_buff PRIVATE openvino::runtime)
elseif(TENSOR_RT_MAKE)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(auto_buff PRIVATE SP_AUTO_BUFF_TENSORRT)
    target_include_directories(auto_buff PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(auto_buff PRIVATE
        ${TENSORRT_INFER_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
        CUDA::cudart)
endif()
