#include "ncnn_api.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ObjectDetectInference.h"

namespace {

std::shared_ptr<YoloNcnn> g_obb_model;
std::shared_ptr<YoloNcnn> g_cls_model;
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

bool validate_flat_data(const float* flat_data, int rows, int cols) {
    return flat_data != nullptr && rows > 0 && cols > 0;
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

extern "C" NCNN_API bool ncnnapi_load_obb_model(const char* param_path,
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

extern "C" NCNN_API bool ncnnapi_load_cls_model(const char* param_path,
                                                const char* bin_path,
                                                int size,
                                                bool use_gpu,
                                                int num_threads) {
    if (!param_path || !bin_path) {
        make_error_json("param_path or bin_path is null");
        return false;
    }

    g_cls_model = YoloNcnn::load_cls(param_path, bin_path, size, use_gpu, num_threads);
    if (!g_cls_model) {
        make_error_json("failed to load classification model");
        return false;
    }

    return true;
}

extern "C" NCNN_API const char* ncnnapi_run_obb(const float* flat_data, int rows, int cols) {
    if (!g_obb_model) {
        return make_error_json("OBB model not loaded");
    }
    if (!validate_flat_data(flat_data, rows, cols)) {
        return make_error_json("invalid flat_data, rows or cols");
    }

    std::vector<HeatmapResult> detections;
    const bool success = g_obb_model->run(detections, flat_data, rows, cols, true, 0.03f);
    return build_result_json(detections, success);
}

extern "C" NCNN_API bool ncnnapi_run_cls(const float* flat_data,
                                         int rows,
                                         int cols,
                                         int* class_id,
                                         float* confidence) {
    if (!g_cls_model) {
        return false;
    }
    if (!validate_flat_data(flat_data, rows, cols) || !class_id || !confidence) {
        return false;
    }

    const ClassifyResult result = g_cls_model->runCls(flat_data, rows, cols, true, 0.03f);
    *class_id = result.classId;
    *confidence = result.confidence;
    return true;
}

extern "C" NCNN_API bool ncnnapi_forward(const float* flat_data,
                                         int rows,
                                         int cols,
                                         int* class_id,
                                         float* confidence) {
    if (!g_obb_model || !g_cls_model) {
        return false;
    }
    if (!validate_flat_data(flat_data, rows, cols) || !class_id || !confidence) {
        return false;
    }

    std::vector<HeatmapResult> obb_output;
    ClassifyResult cls_output;
    const bool success = g_obb_model->forward(g_cls_model, flat_data, rows, cols, obb_output, cls_output, true, 0.03f);
    *class_id = cls_output.classId;
    *confidence = cls_output.confidence;
    build_result_json(obb_output, success);
    return success;
}

extern "C" NCNN_API bool ncnnapi_is_obb_gpu_active() {
    return g_obb_model && g_obb_model->isUsingVulkan();
}

extern "C" NCNN_API bool ncnnapi_is_cls_gpu_active() {
    return g_cls_model && g_cls_model->isUsingVulkan();
}

extern "C" NCNN_API void ncnnapi_release() {
    g_obb_model.reset();
    g_cls_model.reset();
    g_last_json.clear();
}
