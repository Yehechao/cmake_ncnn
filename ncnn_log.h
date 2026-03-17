#pragma once

#if defined(__ANDROID__)
#include <android/log.h>

#define NCNNAPI_LOG_TAG "ncnn_api"
#define NCNNAPI_LOGI(...) __android_log_print(ANDROID_LOG_INFO, NCNNAPI_LOG_TAG, __VA_ARGS__)
#define NCNNAPI_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, NCNNAPI_LOG_TAG, __VA_ARGS__)

#else
#include <cstdio>

#define NCNNAPI_LOGI(...)                  \
    do {                                \
        std::fprintf(stdout, __VA_ARGS__); \
        std::fprintf(stdout, "\n");     \
    } while (0)

#define NCNNAPI_LOGE(...)                  \
    do {                                \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");     \
    } while (0)

#endif
