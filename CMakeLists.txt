cmake_minimum_required(VERSION 3.15)
project(ncnn_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_NCNN_DLL "Build ncnn_api as a shared library" OFF)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(HEADERS
    ObjectDetectInference.h
    utils.h
)

set(COMMON_SOURCES
    YoloInference.cpp
    Post.cpp
    Heatmap.cpp
    Classification.cpp
)
set(EXE_SOURCES
    main.cpp
    ${COMMON_SOURCES}
)
set(DLL_SOURCES
    ncnn_api.cpp
    ${COMMON_SOURCES}
)

if(BUILD_NCNN_DLL)
    add_library(${PROJECT_NAME} SHARED ${DLL_SOURCES} ${HEADERS} ncnn_api.h)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NCNN_API_EXPORTS)
else()
    add_executable(${PROJECT_NAME} ${EXE_SOURCES} ${HEADERS})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ncnn_3rdpart")
    message(FATAL_ERROR "Third-party directory not found: ncnn_3rdpart")
endif()

set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/ncnn-20260113-windows-vs2022/x64)
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/opencv/build)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OPENCV_DIR}/include
    ${NCNN_DIR}/include
    ${NCNN_DIR}/include/ncnn
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${NCNN_DIR}/lib
    ${OPENCV_DIR}/x64/vc16/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ncnn
    glslang
    opencv_world480
)
