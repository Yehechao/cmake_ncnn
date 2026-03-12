#pragma once

#include <string>

#if defined(_WIN32)
#define NCNN_API_EXPORT __declspec(dllexport)
#else
#define NCNN_API_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

NCNN_API_EXPORT bool ncnnapi_load_obb_model(const char* param_path,
                                            const char* bin_path,
                                            int size,
                                            float conf,
                                            float iou,
                                            bool use_gpu,
                                            int num_threads = -1);

NCNN_API_EXPORT const char* ncnnapi_run_obb(const float* flat_data,
                                            int rows,
                                            int cols);

NCNN_API_EXPORT bool isGpuActive();

NCNN_API_EXPORT void ncnnapi_release();

}
