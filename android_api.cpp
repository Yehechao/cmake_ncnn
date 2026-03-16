#include "android_api.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ObjectDetectInference.h"

namespace {

std::shared_ptr<YoloNcnn> g_obb_model;
std::vector<HeatmapResult> g_detection_buffer;

} // namespace

extern "C" NCNN_API_EXPORT bool ncnnapi_load_obb_model(const char* param_path,
                                                       const char* bin_path,
                                                       int size,
                                                       float conf,
                                                       float iou,
                                                       bool use_gpu,
                                                       int num_threads) {
    if (!param_path || !bin_path) {
        return false;
    }

    g_obb_model = YoloNcnn::load_obb(param_path, bin_path, size, conf, iou, use_gpu, num_threads);
    if (!g_obb_model) {
        return false;
    }

    g_detection_buffer.clear();
    g_detection_buffer.reserve(64);
    return true;
}

extern "C" NCNN_API_EXPORT bool ncnnapi_run_obb_struct(const float* flat_data,
                                                       int rows,
                                                       int cols,
                                                       float* out_buffer,
                                                       int max_detections,
                                                       int* out_count) {
    if (out_count) {
        *out_count = 0;
    }

    if (!g_obb_model || !flat_data || rows <= 0 || cols <= 0 || !out_buffer || max_detections <= 0 || !out_count) {
        return false;
    }

    g_detection_buffer.clear();
    const bool success = g_obb_model->run(g_detection_buffer, flat_data, rows, cols, true, 0.03f);
    if (!success) {
        return false;
    }

    const int available = static_cast<int>(g_detection_buffer.size());
    const int written = std::min(available, max_detections);
    *out_count = written;

    for (int i = 0; i < written; ++i) {
        const HeatmapResult& r = g_detection_buffer[static_cast<size_t>(i)];
        const int base = i * NCNNAPI_OBB_FIELDS_PER_DET;
        out_buffer[base + 0] = static_cast<float>(r.id);
        out_buffer[base + 1] = r.confidence;
        out_buffer[base + 2] = r.cx;
        out_buffer[base + 3] = r.cy;
        out_buffer[base + 4] = r.l;
        out_buffer[base + 5] = r.s;
        out_buffer[base + 6] = r.angle;
    }

    return true;
}

extern "C" NCNN_API_EXPORT bool isGpuActive() {
    return g_obb_model && g_obb_model->isUsingVulkan();
}

extern "C" NCNN_API_EXPORT void ncnnapi_release() {
    g_obb_model.reset();
    g_detection_buffer.clear();
    g_detection_buffer.shrink_to_fit();
}
