cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0144 NEW)

project(cmake_ncnn LANGUAGES CXX)

if(NOT MSVC)
    message(FATAL_ERROR "This project currently supports the bundled Windows/MSVC third-party packages only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NCNN_PACKAGE_ROOT "${CMAKE_SOURCE_DIR}/ncnn_3rdpart/ncnn-20260113-windows-vs2022/x64")
set(OpenCV_PACKAGE_ROOT "${CMAKE_SOURCE_DIR}/ncnn_3rdpart/opencv/build")

set(ncnn_DIR "${NCNN_PACKAGE_ROOT}/lib/cmake/ncnn")
set(OpenCV_DIR "${OpenCV_PACKAGE_ROOT}/x64/vc16/lib")

find_package(ncnn CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

add_executable(cmake_ncnn
    Classification.cpp
    Heatmap.cpp
    main.cpp
    Post.cpp
    YoloInference.cpp
    ObjectDetectInference.h
    utils.h
)

target_include_directories(cmake_ncnn PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${NCNN_PACKAGE_ROOT}/include"
    "${OpenCV_PACKAGE_ROOT}/include"
)
target_link_libraries(cmake_ncnn PRIVATE ncnn ${OpenCV_LIBS})

target_compile_options(cmake_ncnn PRIVATE /utf-8 /EHsc)

set_target_properties(cmake_ncnn PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
)
