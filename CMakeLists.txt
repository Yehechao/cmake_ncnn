cmake_minimum_required(VERSION 3.15)
project(ncnn_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

if(ANDROID)
    set(ANDROID_JNI_CLASS "matrix_ncnn/app/NcnnApi" CACHE STRING "JNI class path for RegisterNatives")

    set(SOURCES
        ${COMMON_SOURCES}
        android_api.cpp
        android_jni.cpp
    )

    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS} android_api.h)

    set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/ncnn-20260113-android-vulkan/${ANDROID_ABI})
    set(OPENCV_ANDROID_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/opencv-mobile-4.8.0-android/sdk/native)

    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(ANDROID_LLVM_ARCH aarch64)
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(ANDROID_LLVM_ARCH arm)
    elseif(ANDROID_ABI STREQUAL "x86")
        set(ANDROID_LLVM_ARCH i386)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(ANDROID_LLVM_ARCH x86_64)
    elseif(ANDROID_ABI STREQUAL "riscv64")
        set(ANDROID_LLVM_ARCH riscv64)
    else()
        message(FATAL_ERROR "Unsupported ANDROID_ABI for OpenMP runtime: ${ANDROID_ABI}")
    endif()

    set(NDK_LLVM_LIB_DIR ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows-x86_64/lib/clang/21/lib/linux/${ANDROID_LLVM_ARCH})
    set(ANDROID_OPENMP_LIB ${NDK_LLVM_LIB_DIR}/libomp.a)

    if(NOT EXISTS ${ANDROID_OPENMP_LIB})
        message(FATAL_ERROR "OpenMP runtime not found: ${ANDROID_OPENMP_LIB}")
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${NCNN_DIR}/include
        ${NCNN_DIR}/include/ncnn
        ${OPENCV_ANDROID_DIR}/jni/include
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        NCNN_API_JNI_CLASS_NAME="${ANDROID_JNI_CLASS}"
    )

    find_library(LOG_LIB log)
    find_library(ANDROID_LIB android)
    find_library(VULKAN_LIB vulkan)
    find_library(ATOMIC_LIB atomic)

    set(ANDROID_SYSTEM_LIBS
        ${LOG_LIB}
        ${ANDROID_LIB}
        ${VULKAN_LIB}
    )

    if (ATOMIC_LIB)
        list(APPEND ANDROID_SYSTEM_LIBS ${ATOMIC_LIB})
    endif()

    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${NCNN_DIR}/lib/libncnn.a
        ${NCNN_DIR}/lib/libglslang.a
        ${NCNN_DIR}/lib/libMachineIndependent.a
        ${NCNN_DIR}/lib/libGenericCodeGen.a
        ${NCNN_DIR}/lib/libOSDependent.a
        ${NCNN_DIR}/lib/libSPIRV.a
        ${NCNN_DIR}/lib/libglslang-default-resource-limits.a
        ${OPENCV_ANDROID_DIR}/staticlibs/${ANDROID_ABI}/libopencv_core.a
        ${OPENCV_ANDROID_DIR}/staticlibs/${ANDROID_ABI}/libopencv_imgproc.a
        ${ANDROID_OPENMP_LIB}
        ${ANDROID_SYSTEM_LIBS}
    )
else()
    set(SOURCES
        main.cpp
        ${COMMON_SOURCES}
    )

    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

    # 第三方库根目录
    set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/ncnn-20260113-windows-vs2022/x64)
    set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/opencv/build)

    target_include_directories(${PROJECT_NAME} PRIVATE
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

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/opencv_world480.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/models"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/data"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
    )
endif()
