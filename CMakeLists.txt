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

option(BUILD_CMAKE_NCNN_EXE "Build cmake_ncnn executable" ON)
option(BUILD_CMAKE_NCNN_DLL "Build cmake_ncnn shared library (DLL)" OFF)

if(NOT BUILD_CMAKE_NCNN_EXE AND NOT BUILD_CMAKE_NCNN_DLL)
    message(FATAL_ERROR "At least one target must be enabled: BUILD_CMAKE_NCNN_EXE or BUILD_CMAKE_NCNN_DLL")
endif()

# Release 优化：与 Visual Studio 项目保持一致
# /O2 /Oi /Gy 由 MSVC 默认 Release flags 提供
# /GL + /LTCG 由 WholeProgramOptimization 提供
function(enable_release_optimizations target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:/Oi>
            $<$<CONFIG:Release>:/Gy>
            $<$<CONFIG:Release>:/GL>
        )
        target_link_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
            $<$<CONFIG:Release>:/INCREMENTAL:NO>
        )
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endfunction()

set(CMAKE_NCNN_CORE_SOURCES
    Classification.cpp
    Heatmap.cpp
    Post.cpp
    YoloInference.cpp
)

set(CMAKE_NCNN_PUBLIC_HEADERS
    ObjectDetectInference.h
    utils.h
)

if(BUILD_CMAKE_NCNN_EXE)
    add_executable(cmake_ncnn
        main.cpp
        ${CMAKE_NCNN_CORE_SOURCES}
        ${CMAKE_NCNN_PUBLIC_HEADERS}
    )

    target_include_directories(cmake_ncnn PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${NCNN_PACKAGE_ROOT}/include"
        "${OpenCV_PACKAGE_ROOT}/include"
    )
    target_link_libraries(cmake_ncnn PRIVATE ncnn ${OpenCV_LIBS})
    target_compile_options(cmake_ncnn PRIVATE /utf-8 /EHsc)
    target_compile_definitions(cmake_ncnn PRIVATE NCNN_API_EXPORTS)
    enable_release_optimizations(cmake_ncnn)

    set_target_properties(cmake_ncnn PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    )
endif()

if(BUILD_CMAKE_NCNN_DLL)
    add_library(cmake_ncnn_dll SHARED
        ${CMAKE_NCNN_CORE_SOURCES}
        ${CMAKE_NCNN_PUBLIC_HEADERS}
    )

    target_include_directories(cmake_ncnn_dll PRIVATE
        "${CMAKE_SOURCE_DIR}"
        "${NCNN_PACKAGE_ROOT}/include"
        "${OpenCV_PACKAGE_ROOT}/include"
    )
    target_link_libraries(cmake_ncnn_dll PRIVATE ncnn ${OpenCV_LIBS})
    target_compile_options(cmake_ncnn_dll PRIVATE /utf-8 /EHsc)
    target_compile_definitions(cmake_ncnn_dll PRIVATE NCNN_API_EXPORTS)
    enable_release_optimizations(cmake_ncnn_dll)

    set_target_properties(cmake_ncnn_dll PROPERTIES
        OUTPUT_NAME "cmake_ncnn"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    )
endif()
