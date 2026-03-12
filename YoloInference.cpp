#include "ObjectDetectInference.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
#include <filesystem>
#endif
#if defined(__ANDROID__)
#include <ncnn/gpu.h>
#endif
#include <algorithm>
#include <thread>

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

#if defined(__ANDROID__)
struct VulkanRuntime {
    VulkanRuntime() {
        ncnn::create_gpu_instance();
        gpu_count = ncnn::get_gpu_count();
    }

    ~VulkanRuntime() {
        ncnn::destroy_gpu_instance();
    }

    int gpu_count = 0;
};

VulkanRuntime& getVulkanRuntime() {
    static VulkanRuntime runtime;
    return runtime;
}
#endif
}
 
// YoloNcnn 构造函数
YoloNcnn::YoloNcnn(int netWidth, int netHeight)
    : m_netWidth(netWidth),
    m_netHeight(netHeight),
    m_confidenceThreshold(0.25f),
    m_nmsThreshold(0.45f),
    m_isClassifier(false),
    m_useVulkanCompute(false),
    m_heatmapCols(64) {

    // 初始化精确颜色映射表
    initColormapTableExact();
}


YoloNcnn::~YoloNcnn() {
    m_net.clear();
}

bool YoloNcnn::shouldUseVulkan() {
#if defined(__ANDROID__)
    return getVulkanRuntime().gpu_count > 0;
#else
    return false;
#endif
}


std::shared_ptr<YoloNcnn> YoloNcnn::load_obb(
    const std::string& paramPath, const std::string& binPath,
    int size, float conf, float iou, bool preferGpu, int numThreads) {
    
#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
    // 检查模型文件是否存在
    if (!std::filesystem::exists(paramPath)) {
        std::cerr << "模型param文件不存在: " << paramPath << std::endl;
        return nullptr;
    }
    if (!std::filesystem::exists(binPath)) {
        std::cerr << "模型bin文件不存在: " << binPath << std::endl;
        return nullptr;
    }
#endif

    std::shared_ptr<YoloNcnn> detector(new YoloNcnn(size, size));
    detector->m_confidenceThreshold = conf;
    detector->m_nmsThreshold = iou;

    if (!detector->initialize(paramPath, binPath, preferGpu, numThreads)) {
        std::cerr << "模型初始化失败" << std::endl;
        return nullptr;
    }

    return detector;
}

bool YoloNcnn::initialize(const std::string& paramPath, const std::string& binPath, bool preferGpu, int numThreads) {
    try {
        int num_cores = resolveThreadCount(numThreads);
        m_useVulkanCompute = preferGpu && shouldUseVulkan();

        m_net.opt.num_threads = num_cores;
        m_net.opt.use_vulkan_compute = m_useVulkanCompute;
        m_net.opt.use_fp16_packed = true;      // FP16 打包优化
        m_net.opt.use_fp16_storage = true;     // FP16 存储优化
        m_net.opt.use_fp16_arithmetic = m_useVulkanCompute;
        m_net.opt.use_packing_layout = true;   // 使用打包布局优化
        m_net.opt.lightmode = true;            // 轻量模式，减少内存占用
        
        // 加载模型
        if (m_net.load_param(paramPath.c_str()) != 0) {
            std::cerr << "加载 param 文件失败: " << paramPath << std::endl;
            return false;
        }
        if (m_net.load_model(binPath.c_str()) != 0) {
            std::cerr << "加载 bin 文件失败: " << binPath << std::endl;
            return false;
        }
        
        // 设置输入输出名称（NCNN Ultralytics导出的默认名称）
        m_inputName = "in0";
        m_outputName = "out0";
        
        std::cout << "NCNN OBB模型初始化成功!" << std::endl;
        std::cout << "  输入尺寸: " << m_netWidth << "x" << m_netHeight << std::endl;
        std::cout << "  线程数: " << num_cores
                  << (numThreads > 0 ? " (manual)" : " (auto)") << std::endl;
        std::cout << "  Vulkan: " << (m_useVulkanCompute ? "ON" : "OFF") << std::endl;
        std::cout << "  GPU preference: " << (preferGpu ? "ON" : "OFF") << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "模型初始化错误: " << e.what() << std::endl;
        return false;
    }
}


bool YoloNcnn::runInference(const cv::Mat& inputImg, const float*& outputData, size_t& outputSize) {
    try {
        // 确保输入图像连续
        cv::Mat contImg = inputImg.isContinuous() ? inputImg : inputImg.clone();
        
        const int img_w = contImg.cols;
        const int img_h = contImg.rows;
        
        // 计算 letterbox 缩放比例
        float scale = std::min((float)m_netWidth / img_w, (float)m_netHeight / img_h);
        int new_w = std::max(1, static_cast<int>(img_w * scale + 0.5f));
        int new_h = std::max(1, static_cast<int>(img_h * scale + 0.5f));

        // 使用 OpenCV resize + padding，保持与分类预处理一致的行为
        cv::Mat resized;
        cv::resize(contImg, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        int wpad = m_netWidth - new_w;
        int hpad = m_netHeight - new_h;
        int top = hpad / 2;
        int bottom = hpad - top;
        int left = wpad / 2;
        int right = wpad - left;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        cv::Mat paddedContiguous = padded.isContinuous() ? padded : padded.clone();

        ncnn::Mat in_pad = ncnn::Mat::from_pixels(
            paddedContiguous.data,
            ncnn::Mat::PIXEL_BGR2RGB,
            m_netWidth,
            m_netHeight
        );
        
        // 归一化：减去 0 均值，乘以 1/255 归一化系数
        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        in_pad.substract_mean_normalize(0, norm_vals);
        
        // 创建 Extractor 执行推理
        ncnn::Extractor ex = m_net.create_extractor();
        ex.set_light_mode(true);   // 轻量模式，减少中间层内存
        
        // 设置输入
        ex.input(m_inputName.c_str(), in_pad);
        
        // 获取输出
        ncnn::Mat out;
        ex.extract(m_outputName.c_str(), out);
        
        // 更新输出形状
        m_outputShape = {1, out.h, out.w};
        
        // 保存 ncnn::Mat 对象，避免数据拷贝
        m_outputMat = out;
        
        // 直接使用 ncnn::Mat 数据指针
        outputData = (const float*)m_outputMat.data;
        outputSize = m_outputMat.total();
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "推理过程中发生错误: " << e.what() << std::endl;
        return false;
    }
}

bool YoloNcnn::run(std::vector<HeatmapResult>& output,
    const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold) {
    output.clear();

    if (heatmapData2D.empty()) {
        return false;
    }

    m_heatmapCols = static_cast<int>(heatmapData2D[0].size());
    cv::Mat heatmapImage = processHeatmapData(heatmapData2D, denoise, threshold);

    float scale = std::min((float)m_netWidth / heatmapImage.cols,
                           (float)m_netHeight / heatmapImage.rows);
    int new_w = std::max(1, static_cast<int>(heatmapImage.cols * scale + 0.5f));
    int new_h = std::max(1, static_cast<int>(heatmapImage.rows * scale + 0.5f));
    int wpad = m_netWidth - new_w;
    int hpad = m_netHeight - new_h;
    cv::Vec4d param(scale, scale, wpad / 2, hpad / 2);

    const float* outputData = nullptr;
    size_t outputSize = 0;
    if (!runInference(heatmapImage, outputData, outputSize)) {
        return false;
    }

    if (outputData == nullptr || outputSize == 0) {
        std::cerr << "推理结果为空" << std::endl;
        return false;
    }

    postprocessHeatmap(output, const_cast<float*>(outputData), param);
    return !output.empty();
}

// ========== 融合推理：OBB检测 + 姿势分类 ==========
bool YoloNcnn::forward(
    std::shared_ptr<YoloNcnn> clsModel,
    const std::vector<std::vector<float>>& heatmapData2D,
    std::vector<HeatmapResult>& obbOutput,
    ClassifyResult& clsOutput,
    bool denoise, float threshold) {
    const bool obbSuccess = run(obbOutput, heatmapData2D, denoise, threshold);

    if (clsModel && clsModel->m_isClassifier) {
        clsOutput = clsModel->runCls(heatmapData2D, denoise, threshold);
    } else {
        clsOutput = ClassifyResult();
    }

    return obbSuccess;
}
