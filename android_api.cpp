#include "android_api.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ObjectDetectInference.h"

namespace {

std::shared_ptr<YoloNcnn> g_obb_model;
std::string g_last_json;

std::string escape_json_string(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char ch : input) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += ch; break;
        }
    }
    return out;
}

const char* make_error_json(const std::string& msg) {
    g_last_json = std::string("{\"success\":false,\"error\":\"")
        + escape_json_string(msg) + "\"}";
    return g_last_json.c_str();
}

std::vector<std::vector<float>> flat_to_2d(const float* flat_data, int rows, int cols) {
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols, 0.0f));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            data[r][c] = flat_data[r * cols + c];
        }
    }
    return data;
}

const char* build_result_json(const std::vector<HeatmapResult>& detections, bool success) {
    std::ostringstream oss;
    oss << "{\"success\":" << (success ? "true" : "false") << ",\"detections\":[";

    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }

        const HeatmapResult& r = detections[i];
        oss << "{\"id\":" << r.id
            << ",\"confidence\":" << r.confidence
            << ",\"cx\":" << r.cx
            << ",\"cy\":" << r.cy
            << ",\"l\":" << r.l
            << ",\"s\":" << r.s
            << ",\"angle\":" << r.angle
            << "}";
    }

    oss << "]}";
    g_last_json = oss.str();
    return g_last_json.c_str();
}

} // namespace

extern "C" NCNN_API_EXPORT bool ncnnapi_load_obb_model(const char* param_path,
                                                       const char* bin_path,
                                                       int size,
                                                       float conf,
                                                       float iou,
                                                       bool use_gpu,
                                                       int num_threads) {
    if (!param_path || !bin_path) {
        make_error_json("param_path or bin_path is null");
        return false;
    }

    g_obb_model = YoloNcnn::load_obb(param_path, bin_path, size, conf, iou, use_gpu, num_threads);
    if (!g_obb_model) {
        make_error_json("failed to load OBB model");
        return false;
    }

    return true;
}

extern "C" NCNN_API_EXPORT const char* ncnnapi_run_obb(const float* flat_data,
                                                       int rows,
                                                       int cols) {
    if (!g_obb_model) {
        return make_error_json("OBB model not loaded");
    }

    if (!flat_data) {
        return make_error_json("flat_data is null");
    }

    if (rows <= 0 || cols <= 0) {
        return make_error_json("invalid rows or cols");
    }

    std::vector<std::vector<float>> heatmap_data = flat_to_2d(flat_data, rows, cols);
    std::vector<HeatmapResult> detections;
    const bool success = g_obb_model->run(detections, heatmap_data, true, 0.03f);
    return build_result_json(detections, success);
}

extern "C" NCNN_API_EXPORT bool isGpuActive() {
    return g_obb_model && g_obb_model->isUsingVulkan();
}

extern "C" NCNN_API_EXPORT void ncnnapi_release() {
    g_obb_model.reset();
    g_last_json.clear();
}
