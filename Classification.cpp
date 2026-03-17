#include "ObjectDetectInference.h"
#include "ncnn_log.h"
#include <thread>
#include <algorithm>
#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
#include <filesystem>
#endif

namespace {
int getLogicalCoreCount() {
    unsigned int logicalCores = std::thread::hardware_concurrency();
    if (logicalCores == 0) {
        logicalCores = 4;
    }
    return static_cast<int>(logicalCores);
}

int getRecommendedThreadCount() {
    const int logicalCores = getLogicalCoreCount();
    int threads = logicalCores / 2;
    if (threads < 1) {
        threads = 1;
    }
    return threads;
}

int resolveThreadCount(int requestedThreads) {
    const int maxThreads = getLogicalCoreCount();
    if (requestedThreads <= 0) {
        return std::min(getRecommendedThreadCount(), maxThreads);
    }
    return std::min(maxThreads, std::max(1, requestedThreads));
}
}

std::shared_ptr<YoloNcnn> YoloNcnn::load_cls(
    const std::string& paramPath, const std::string& binPath,
    int size, bool preferGpu, int numThreads) {

#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
    if (!std::filesystem::exists(paramPath)) {
        NCNNAPI_LOGE("classification param file not found: %s", paramPath.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(binPath)) {
        NCNNAPI_LOGE("classification bin file not found: %s", binPath.c_str());
        return nullptr;
    }
#endif

    std::shared_ptr<YoloNcnn> classifier(new YoloNcnn(size, size));
    classifier->m_isClassifier = true;

    if (!classifier->initializeCls(paramPath, binPath, preferGpu, numThreads)) {
        NCNNAPI_LOGE("classification model initialize failed");
        return nullptr;
    }

    return classifier;
}

bool YoloNcnn::initializeCls(const std::string& paramPath, const std::string& binPath, bool preferGpu, int numThreads) {
    try {
        int num_cores = resolveThreadCount(numThreads);
        m_useVulkanCompute = preferGpu && shouldUseVulkan();

        m_net.opt.num_threads = num_cores;
        m_net.opt.use_vulkan_compute = m_useVulkanCompute;
        m_net.opt.use_fp16_packed = true;
        m_net.opt.use_fp16_storage = true;
        m_net.opt.use_fp16_arithmetic = m_useVulkanCompute;
        m_net.opt.use_packing_layout = true;
        m_net.opt.lightmode = true;

        if (m_net.load_param(paramPath.c_str()) != 0) {
            NCNNAPI_LOGE("load classification param failed: %s", paramPath.c_str());
            return false;
        }
        if (m_net.load_model(binPath.c_str()) != 0) {
            NCNNAPI_LOGE("load classification bin failed: %s", binPath.c_str());
            return false;
        }

        m_inputName = "in0";
        m_outputName = "out0";
        initColormapTableExact();

        NCNNAPI_LOGI("NCNN classification model initialized.");
        NCNNAPI_LOGI("  input size: %dx%d", m_netWidth, m_netHeight);
        NCNNAPI_LOGI("  threads: %d (%s)", num_cores, numThreads > 0 ? "manual" : "auto");
        NCNNAPI_LOGI("  Vulkan: %s", m_useVulkanCompute ? "ON" : "OFF");
        NCNNAPI_LOGI("  GPU preference: %s", preferGpu ? "ON" : "OFF");

        return true;
    }
    catch (const std::exception& e) {
        NCNNAPI_LOGE("classification model init exception: %s", e.what());
        return false;
    }
}

ClassifyResult YoloNcnn::runCls(const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold) {

    ClassifyResult result;

    if (!m_isClassifier) {
        NCNNAPI_LOGE("current model is not classifier, please load with load_cls");
        return result;
    }

    if (heatmapData2D.empty()) {
        return result;
    }

    cv::Mat heatmapImage = processHeatmapData(heatmapData2D, denoise, threshold, 5);
    if (heatmapImage.empty()) {
        NCNNAPI_LOGE("classification inference: heatmap preprocessing failed");
        return result;
    }

    try {
        cv::Mat contImg = heatmapImage.isContinuous() ? heatmapImage : heatmapImage.clone();

        const int img_w = contImg.cols;
        const int img_h = contImg.rows;

        int resized_w = m_netWidth;
        int resized_h = m_netHeight;
        if (img_w > 0 && img_h > 0) {
            if (img_w < img_h) {
                const float scale = static_cast<float>(m_netWidth) / static_cast<float>(img_w);
                resized_w = m_netWidth;
                resized_h = std::max(1, static_cast<int>(img_h * scale + 0.5f));
            }
            else {
                const float scale = static_cast<float>(m_netHeight) / static_cast<float>(img_h);
                resized_h = m_netHeight;
                resized_w = std::max(1, static_cast<int>(img_w * scale + 0.5f));
            }
        }

        cv::Mat resized;
        cv::resize(contImg, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

        const int crop_x = std::max(0, (resized.cols - m_netWidth) / 2);
        const int crop_y = std::max(0, (resized.rows - m_netHeight) / 2);
        const int crop_w = std::min(m_netWidth, resized.cols - crop_x);
        const int crop_h = std::min(m_netHeight, resized.rows - crop_y);
        cv::Mat cropped = resized(cv::Rect(crop_x, crop_y, crop_w, crop_h));
        if (cropped.cols != m_netWidth || cropped.rows != m_netHeight) {
            cv::resize(cropped, cropped, cv::Size(m_netWidth, m_netHeight), 0, 0, cv::INTER_LINEAR);
        }
        cv::Mat croppedContiguous = cropped.isContinuous() ? cropped : cropped.clone();

        ncnn::Mat in = ncnn::Mat::from_pixels(
            croppedContiguous.data,
            ncnn::Mat::PIXEL_BGR2RGB,
            m_netWidth,
            m_netHeight
        );

        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = m_net.create_extractor();
        ex.set_light_mode(true);

        ex.input(m_inputName.c_str(), in);

        ncnn::Mat out;
        ex.extract(m_outputName.c_str(), out);

        if (out.empty()) {
            NCNNAPI_LOGE("classification inference error: empty output tensor");
            return result;
        }

        ncnn::Mat out_flattened = out.reshape(out.w * out.h * out.c);
        size_t numClasses = out_flattened.w;

        if (numClasses == 0) {
            NCNNAPI_LOGE("classification inference error: class count is zero");
            return result;
        }

        int bestClassId = 0;
        float bestScore = out_flattened[0];

        for (size_t i = 1; i < numClasses; ++i) {
            if (out_flattened[i] > bestScore) {
                bestScore = out_flattened[i];
                bestClassId = static_cast<int>(i);
            }
        }

        result.classId = bestClassId;
        result.confidence = bestScore;
    }
    catch (const std::exception& e) {
        NCNNAPI_LOGE("classification inference exception: %s", e.what());
    }

    return result;
}
