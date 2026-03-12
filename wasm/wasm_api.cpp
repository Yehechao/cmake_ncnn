#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ObjectDetectInference.h"

using namespace emscripten;

// 仅保留 OBB 模型
static std::shared_ptr<YoloNcnn> g_obbModel = nullptr;

// 最近一次缓存
static std::vector<std::vector<float>> g_lastHeatmapData2D;
static std::vector<HeatmapResult> g_lastDetections;
static std::string g_lastResultJson;

static std::string escapeJsonString(const std::string& input) {
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

static std::string makeErrorJson(const std::string& msg) {
    g_lastResultJson = std::string("{\"success\":false,\"error\":\"") + escapeJsonString(msg) + "\"}";
    return g_lastResultJson;
}

static bool validateInput(const val& flatData, int rows, int cols, std::string& errMsg) {
    if (rows <= 0 || cols <= 0) {
        errMsg = "Invalid rows/cols";
        return false;
    }

    const unsigned int len = flatData["length"].as<unsigned int>();
    const unsigned int required = static_cast<unsigned int>(rows * cols);
    if (len < required) {
        errMsg = "Input data length is smaller than rows*cols";
        return false;
    }

    return true;
}

static std::vector<std::vector<float>> jsArrayTo2D(const val& jsData, int rows, int cols) {
    const unsigned int required = static_cast<unsigned int>(rows * cols);
    std::vector<float> flat(required, 0.0f);

    val wasmView = val(typed_memory_view(required, flat.data()));
    wasmView.call<void>("set", jsData);

    std::vector<std::vector<float>> data(rows, std::vector<float>(cols, 0.0f));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            data[r][c] = flat[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
        }
    }
    return data;
}

static void appendDetectionsArray(
    std::ostringstream& oss,
    const std::vector<HeatmapResult>& results) {

    oss << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) oss << ",";

        const HeatmapResult& r = results[i];
        oss << "{\"id\":" << r.id
            << ",\"confidence\":" << r.confidence
            << ",\"cx\":" << r.cx
            << ",\"cy\":" << r.cy
            << ",\"l\":" << r.l
            << ",\"s\":" << r.s
            << ",\"angle\":" << r.angle
            << "}";
    }
    oss << "]";
}

static std::string buildResultJson(bool success) {
    std::ostringstream oss;
    oss << "{\"success\":" << (success ? "true" : "false")
        << ",\"detections\":";

    appendDetectionsArray(oss, g_lastDetections);
    oss << "}";

    g_lastResultJson = oss.str();
    return g_lastResultJson;
}

bool loadObbModel(const std::string& paramPath, const std::string& binPath, int size, float conf, float iou) {
    g_obbModel = YoloNcnn::load_obb(paramPath, binPath, size, conf, iou);
    if (!g_obbModel) {
        std::cerr << "[WASM] failed to load OBB model" << std::endl;
        return false;
    }

    std::cout << "[WASM] OBB model loaded, size=" << size << std::endl;
    return true;
}

std::string runObb(const val& flatData, int rows, int cols) {
    if (!g_obbModel) {
        return makeErrorJson("OBB model not loaded");
    }

    std::string errMsg;
    if (!validateInput(flatData, rows, cols, errMsg)) {
        return makeErrorJson(errMsg);
    }

    g_lastHeatmapData2D = jsArrayTo2D(flatData, rows, cols);
    g_lastDetections.clear();

    const bool success = g_obbModel->run(g_lastDetections, g_lastHeatmapData2D, true, 0.03f);

    return buildResultJson(success);
}

EMSCRIPTEN_BINDINGS(yolo_ncnn_wasm) {
    // 仅保留加载模型与推理接口
    function("loadObbModel", &loadObbModel);
    function("runObb", &runObb);
}
