cmake_minimum_required(VERSION 3.15)
project(ncnn_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ============ 源文件
set(SOURCES
    main.cpp
    YoloInference.cpp
    Post.cpp
    Heatmap.cpp
    Classification.cpp
)

set(HEADERS
    ObjectDetectInference.h
    utils.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#第三方库根目录
set(NCNN_DIR  ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/ncnn-20260113-windows-vs2022/x64)
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/ncnn_3rdpart/opencv/build)

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OPENCV_DIR}/include
    ${NCNN_DIR}/include
    ${NCNN_DIR}/include/ncnn
)

# 库目录
target_link_directories(${PROJECT_NAME} PRIVATE
    ${NCNN_DIR}/lib
    ${OPENCV_DIR}/x64/vc16/lib
)

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ncnn
    glslang
    opencv_world480
)

# 自动复制 dll 和 models
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
