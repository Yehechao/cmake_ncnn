#pragma once

#if defined(_WIN32)
#if defined(NCNN_API_EXPORTS)
#define NCNN_API __declspec(dllexport)
#else
#define NCNN_API __declspec(dllimport)
#endif
#else
#define NCNN_API
#endif

extern "C" {

NCNN_API bool ncnnapi_load_obb_model(const char* param_path,
                                     const char* bin_path,
                                     int size,
                                     float conf,
                                     float iou,
                                     bool use_gpu,
                                     int num_threads);

NCNN_API bool ncnnapi_load_cls_model(const char* param_path,
                                     const char* bin_path,
                                     int size,
                                     bool use_gpu,
                                     int num_threads);

NCNN_API const char* ncnnapi_run_obb(const float* flat_data,
                                     int rows,
                                     int cols);

NCNN_API bool ncnnapi_run_cls(const float* flat_data,
                              int rows,
                              int cols,
                              int* class_id,
                              float* confidence);

NCNN_API bool ncnnapi_forward(const float* flat_data,
                              int rows,
                              int cols,
                              int* class_id,
                              float* confidence);

NCNN_API bool ncnnapi_is_obb_gpu_active();
NCNN_API bool ncnnapi_is_cls_gpu_active();
NCNN_API void ncnnapi_release();

}
