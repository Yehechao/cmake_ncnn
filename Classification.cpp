#include "ObjectDetectInference.h"
#include <iostream>
#include <thread>
#include <algorithm>
#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
#include <filesystem>
#endif

namespace {
int getRecommendedThreadCount() {
    unsigned int logicalCores = std::thread::hardware_concurrency();
    if (logicalCores == 0) {
        logicalCores = 4;
    }

    int threads = static_cast<int>(logicalCores / 2);
    if (threads < 1) {
        threads = 1;
    }
    return threads;
}

int resolveThreadCount(int requestedThreads) {
    if (requestedThreads <= 0) {
        return getRecommendedThreadCount();
    }
    return std::max(1, requestedThreads);
}
}

std::shared_ptr<YoloNcnn> YoloNcnn::load_cls(
    const std::string& paramPath, const std::string& binPath,
    int size, bool preferGpu, int numThreads) {

#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
    if (!std::filesystem::exists(paramPath)) {
        std::cerr << "classification param file not found: " << paramPath << std::endl;
        return nullptr;
    }
    if (!std::filesystem::exists(binPath)) {
        std::cerr << "classification bin file not found: " << binPath << std::endl;
        return nullptr;
    }
#endif

    std::shared_ptr<YoloNcnn> classifier(new YoloNcnn(size, size));
    classifier->m_isClassifier = true;

    if (!classifier->initializeCls(paramPath, binPath, preferGpu, numThreads)) {
        std::cerr << "classification model initialize failed" << std::endl;
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
            std::cerr << "load classification param failed: " << paramPath << std::endl;
            return false;
        }
        if (m_net.load_model(binPath.c_str()) != 0) {
            std::cerr << "load classification bin failed: " << binPath << std::endl;
            return false;
        }

        m_inputName = "in0";
        m_outputName = "out0";
        initColormapTableExact();

        std::cout << "NCNN classification model initialized." << std::endl;
        std::cout << "  input size: " << m_netWidth << "x" << m_netHeight << std::endl;
        std::cout << "  threads: " << num_cores
                  << (numThreads > 0 ? " (manual)" : " (auto)") << std::endl;
        std::cout << "  Vulkan: " << (m_useVulkanCompute ? "ON" : "OFF") << std::endl;
        std::cout << "  GPU preference: " << (preferGpu ? "ON" : "OFF") << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "classification model init exception: " << e.what() << std::endl;
        return false;
    }
}

ClassifyResult YoloNcnn::runCls(const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold) {

    ClassifyResult result;

    if (!m_isClassifier) {
        std::cerr << "current model is not classifier, please load with load_cls" << std::endl;
        return result;
    }

    if (heatmapData2D.empty()) {
        return result;
    }

    cv::Mat heatmapImage = processHeatmapData(heatmapData2D, denoise, threshold, 5);
    if (heatmapImage.empty()) {
        std::cerr << "classification inference: heatmap preprocessing failed" << std::endl;
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
            std::cerr << "classification inference error: empty output tensor" << std::endl;
            return result;
        }

        ncnn::Mat out_flattened = out.reshape(out.w * out.h * out.c);
        size_t numClasses = out_flattened.w;

        if (numClasses == 0) {
            std::cerr << "classification inference error: class count is zero" << std::endl;
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
        std::cerr << "classification inference exception: " << e.what() << std::endl;
    }

    return result;
}

ClassifyResult YoloNcnn::runCls(const float* flatData,
    int rows,
    int cols,
    bool denoise,
    float threshold) {

    ClassifyResult result;

    if (!m_isClassifier) {
        std::cerr << "current model is not classifier, please load with load_cls" << std::endl;
        return result;
    }

    if (!flatData || rows <= 0 || cols <= 0) {
        return result;
    }

    cv::Mat heatmapImage = processHeatmapData(flatData, rows, cols, denoise, threshold, 5);
    if (heatmapImage.empty()) {
        std::cerr << "classification inference: heatmap preprocessing failed" << std::endl;
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
            std::cerr << "classification inference error: empty output tensor" << std::endl;
            return result;
        }

        ncnn::Mat out_flattened = out.reshape(out.w * out.h * out.c);
        size_t numClasses = out_flattened.w;

        if (numClasses == 0) {
            std::cerr << "classification inference error: class count is zero" << std::endl;
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
        std::cerr << "classification inference exception: " << e.what() << std::endl;
    }

    return result;
}
