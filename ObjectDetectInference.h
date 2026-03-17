// ObjectDetectInference.h
#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

class YoloNcnn {
public:
    // 加载 OBB 检测模型
    static std::shared_ptr<YoloNcnn> load_obb(
        const std::string& paramPath,
        const std::string& binPath,
        int size,
        float conf = 0.25f,
        float iou = 0.45f,
        bool preferGpu = false,
        int numThreads = -1);

    // 加载分类模型
    static std::shared_ptr<YoloNcnn> load_cls(
        const std::string& paramPath,
        const std::string& binPath,
        int size,
        bool preferGpu = false,
        int numThreads = -1);

    ~YoloNcnn();

    // OBB 推理：仅返回一份检测结果（更轻量）
    bool run(std::vector<HeatmapResult>& output,
             const std::vector<std::vector<float>>& heatmapData2D,
             bool denoise = true,
             float threshold = 0.03f);
    bool run(std::vector<HeatmapResult>& output,
             const float* flatData,
             int rows,
             int cols,
             bool denoise = true,
             float threshold = 0.03f);

    // 获取网络信息
    int getNetWidth() const { return m_netWidth; }
    int getNetHeight() const { return m_netHeight; }
    bool isUsingVulkan() const { return m_useVulkanCompute; }

    // 设置阈值
    void setConfidenceThreshold(float conf) { m_confidenceThreshold = conf; }
    void setNMSThreshold(float iou) { m_nmsThreshold = iou; }

    // 融合推理兼容接口：仅返回 OBB + 分类
    bool forward(std::shared_ptr<YoloNcnn> clsModel,
                 const std::vector<std::vector<float>>& heatmapData2D,
                 std::vector<HeatmapResult>& obbOutput,
                 ClassifyResult& clsOutput,
                 bool denoise = true,
                 float threshold = 0.03f);

    // 分类推理
    ClassifyResult runCls(const std::vector<std::vector<float>>& heatmapData2D,
                          bool denoise = true,
                          float threshold = 0.03f);

private:
    YoloNcnn(int netWidth = 640, int netHeight = 640);
    YoloNcnn(const YoloNcnn&) = delete;
    YoloNcnn& operator=(const YoloNcnn&) = delete;

    // 初始化
    bool initialize(const std::string& paramPath, const std::string& binPath, bool preferGpu, int numThreads);
    bool initializeCls(const std::string& paramPath, const std::string& binPath, bool preferGpu, int numThreads);
    static bool shouldUseVulkan();

    // 通用推理执行（直接返回输出张量指针，避免拷贝）
    bool runInference(const cv::Mat& inputImg, const float*& outputData, size_t& outputSize);

    // 后处理（热力图）
    void postprocessHeatmap(std::vector<HeatmapResult>& output,
                            float* data,
                            const cv::Vec4d& param);

    // 热力图处理（scale: OBB=10, Cls=5）
    cv::Mat processHeatmapData(const std::vector<std::vector<float>>& heatmapData2D,
                               bool denoise = true,
                               float threshold = 0.03f,
                               int scale = 10);
    cv::Mat processHeatmapData(const float* flatData,
                               int rows,
                               int cols,
                               bool denoise = true,
                               float threshold = 0.03f,
                               int scale = 10);

    // 公共热力图预处理（数据转换+去噪）
    cv::Mat prepareHeatmap(const std::vector<std::vector<float>>& heatmapData2D,
                           bool denoise = true,
                           float threshold = 0.03f);
    cv::Mat prepareHeatmap(const float* flatData,
                           int rows,
                           int cols,
                           bool denoise = true,
                           float threshold = 0.03f);

    void initColormapTableExact();

    static float calc_rotate_iou(const cv::RotatedRect& rect1,
                                 const cv::RotatedRect& rect2);

    void filterMaxOnePerClassHeatmap(std::vector<HeatmapResult>& results);
    bool isLimitedClass(int classId) const;

private:
    // 网络参数
    int m_netWidth;
    int m_netHeight;
    float m_confidenceThreshold;
    float m_nmsThreshold;
    bool m_isClassifier;
    bool m_useVulkanCompute;

    // NCNN
    ncnn::Net m_net;

    // 输入输出信息
    std::string m_inputName;
    std::string m_outputName;
    std::vector<int> m_outputShape;

    // 推理输出
    ncnn::Mat m_outputMat;

    // 预处理复用缓冲（减少每帧 Mat 分配）
    cv::Mat m_heatmapFloatBuffer;
    cv::Mat m_heatmapNormalizedBuffer;
    cv::Mat m_heatmap8UBuffer;
    cv::Mat m_heatmapColorBuffer;
    cv::Mat m_heatmapEnlargedBuffer;

    // 颜色映射表
    cv::Mat m_colormapTable;

    // 当前热力图列数（32 或 64）
    int m_heatmapCols;
};
