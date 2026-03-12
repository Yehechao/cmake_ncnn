cmake_minimum_required(VERSION 3.15)
project(yolo_ncnn_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 是否生成 Embind 的 TypeScript 声明文件（.d.ts）
option(ENABLE_EMIT_TSD "Emit TypeScript declaration for embind exports" ON)
# 是否启用 wasm SIMD
option(WASM_USE_SIMD "Enable wasm SIMD (-msimd128)" OFF)
# 是否启用 wasm pthread 多线程
option(WASM_USE_PTHREADS "Enable wasm pthreads" OFF)
# pthread 线程池大小
set(WASM_PTHREAD_POOL_SIZE 8 CACHE STRING "Emscripten pthread pool size")

# 若环境没有 tsc，则自动关闭 --emit-tsd，避免链接报错
if (ENABLE_EMIT_TSD)
    find_program(TSC_EXECUTABLE NAMES tsc)
    if (NOT TSC_EXECUTABLE)
        message(WARNING "ENABLE_EMIT_TSD=ON but 'tsc' not found. Disable --emit-tsd for this build.")
        set(ENABLE_EMIT_TSD OFF CACHE BOOL "Emit TypeScript declaration for embind exports" FORCE)
    endif()
endif()

set(SRC_DIR ${CMAKE_SOURCE_DIR}/..)

set(SOURCES
    ${SRC_DIR}/YoloInference.cpp
    ${SRC_DIR}/Post.cpp
    ${SRC_DIR}/Heatmap.cpp
    ${SRC_DIR}/Classification.cpp
    wasm_api.cpp
)

set(HEADERS
    ${SRC_DIR}/ObjectDetectInference.h
    ${SRC_DIR}/utils.h
)

set(NCNN_WASM_ROOT ${SRC_DIR}/ncnn_3rdpart/ncnn-20260113-webassembly)
if (WASM_USE_SIMD AND WASM_USE_PTHREADS)
    set(NCNN_WASM_VARIANT "simd-threads")
elseif (WASM_USE_SIMD AND NOT WASM_USE_PTHREADS)
    set(NCNN_WASM_VARIANT "simd")
elseif (NOT WASM_USE_SIMD AND WASM_USE_PTHREADS)
    set(NCNN_WASM_VARIANT "threads")
else()
    set(NCNN_WASM_VARIANT "basic")
endif()
set(NCNN_WASM_DIR ${NCNN_WASM_ROOT}/${NCNN_WASM_VARIANT})
set(OPENCV_WASM_DIR ${SRC_DIR}/ncnn_3rdpart/opencv_wasm_install)

message(STATUS "NCNN wasm variant: ${NCNN_WASM_VARIANT}")
message(STATUS "WASM_USE_SIMD=${WASM_USE_SIMD}, WASM_USE_PTHREADS=${WASM_USE_PTHREADS}")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
    ${OPENCV_WASM_DIR}/include/opencv4
    ${NCNN_WASM_DIR}/include
    ${NCNN_WASM_DIR}/include/ncnn
)

# 链接 ncnn + opencv 静态库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${NCNN_WASM_DIR}/lib/libncnn.a
    ${OPENCV_WASM_DIR}/lib/libopencv_imgproc.a
    ${OPENCV_WASM_DIR}/lib/libopencv_imgcodecs.a
    ${OPENCV_WASM_DIR}/lib/libopencv_core.a
)

# 链接 opencv 安装目录中的三方依赖（zlib / libpng / libjpeg-turbo）
set(OPENCV_3RDPARTY_DIR ${OPENCV_WASM_DIR}/lib/opencv4/3rdparty)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENCV_3RDPARTY_DIR}/liblibpng.a
    ${OPENCV_3RDPARTY_DIR}/liblibjpeg-turbo.a
    ${OPENCV_3RDPARTY_DIR}/libzlib.a
)

# 编译选项说明：
# -msimd128: 启用 wasm SIMD
# -pthread: 启用 pthread 多线程
# -O3: 最大化优化
# -DNDEBUG: Release 模式关闭断言
set(WASM_COMPILE_FLAGS
    -O3
    -DNDEBUG
)
if (WASM_USE_SIMD)
    list(APPEND WASM_COMPILE_FLAGS -msimd128)
endif()
if (WASM_USE_PTHREADS)
    list(APPEND WASM_COMPILE_FLAGS -pthread)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE ${WASM_COMPILE_FLAGS})

# ALLOW_MEMORY_GROWTH: 内存可增长
# MAXIMUM_MEMORY / INITIAL_MEMORY: wasm 内存上限和初始值
# USE_PTHREADS / PTHREAD_POOL_SIZE: 多线程开关与线程池大小
# EXPORTED_RUNTIME_METHODS: 导出的运行时方法（FS/ccall/cwrap）
# STACK_SIZE: 栈大小（字节）
if (ENABLE_EMIT_TSD)
    # 生成与 yolo_ncnn_wasm.js 同目录的 yolo_ncnn_wasm.d.ts
    set(EMIT_TSD_FLAG "--emit-tsd ${PROJECT_NAME}.d.ts")
else()
    set(EMIT_TSD_FLAG "")
endif()

set(WASM_LINK_FLAGS
    --bind
    ${EMIT_TSD_FLAG}
    -s WASM=1
    -s ALLOW_MEMORY_GROWTH=1
    -s MAXIMUM_MEMORY=1073741824
    -s INITIAL_MEMORY=268435456
    -s EXPORTED_RUNTIME_METHODS=['FS','ccall','cwrap']
    -s FORCE_FILESYSTEM=1
    -s EXIT_RUNTIME=0
    -s NO_EXIT_RUNTIME=1
    -s ASSERTIONS=0
    -s STACK_SIZE=5242880
    -O3
    -Wl,--allow-multiple-definition
)
if (WASM_USE_SIMD)
    list(APPEND WASM_LINK_FLAGS -msimd128)
endif()
if (WASM_USE_PTHREADS)
    list(APPEND WASM_LINK_FLAGS
        -s USE_PTHREADS=1
        -s PTHREAD_POOL_SIZE=${WASM_PTHREAD_POOL_SIZE}
        -pthread
    )
endif()
string(JOIN " " WASM_LINK_FLAGS_STR ${WASM_LINK_FLAGS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS "${WASM_LINK_FLAGS_STR}"
)

# 安网页端
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/dist)
install(FILES
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}.wasm
    DESTINATION ${CMAKE_SOURCE_DIR}/dist
)
install(FILES
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}.worker.js
    DESTINATION ${CMAKE_SOURCE_DIR}/dist
    OPTIONAL
)
if (ENABLE_EMIT_TSD)
    install(FILES
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}.d.ts
        DESTINATION ${CMAKE_SOURCE_DIR}/dist
        OPTIONAL
    )
endif()

if (EXISTS ${CMAKE_SOURCE_DIR}/index.html)
    install(FILES
        ${CMAKE_SOURCE_DIR}/index.html
        DESTINATION ${CMAKE_SOURCE_DIR}/dist
    )
endif()
