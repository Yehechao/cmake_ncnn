// ObjectDetectInference.h
#pragma once
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include "utils.h"


class NCNN_API YoloNcnn {
public:
    // 加载 OBB 检测模型
    static std::shared_ptr<YoloNcnn> load_obb(
        const std::string& paramPath,
        const std::string& binPath,
        int size,
        float conf = 0.25f,
        float iou = 0.45f,
        int numThreads = 4);

    // 加载分类模型
    static std::shared_ptr<YoloNcnn> load_cls(
        const std::string& paramPath,
        const std::string& binPath,
        int size,
        int numThreads = 4);

    ~YoloNcnn();

    // OBB 推理：仅返回一份检测结果（更轻量）
    bool run(std::vector<HeatmapResult>& output,
        const std::vector<std::vector<float>>& heatmapData2D,
        bool denoise = true,
        float threshold = 0.03f);

    // 绘制热力图检测结果
    void drawPredOnHeatmap(cv::Mat& img, const std::vector<HeatmapResult>& result);

    // 绘制热力图检测结果（带轮廓交集）
    void drawPredOnHeatmap(cv::Mat& img,
        const std::vector<HeatmapResult>& result,
        const std::vector<std::vector<cv::Point2f>>& contours);

    // 绘制热力图检测结果（带数值网格背景，无轮廓）
    void drawPredOnHeatmap(cv::Mat& img,
        const std::vector<HeatmapResult>& result,
        const std::vector<std::vector<float>>& heatmapData2D);

    // 绘制热力图检测结果（带数值网格背景 + 轮廓交集）
    void drawPredOnHeatmap(cv::Mat& img,
        const std::vector<HeatmapResult>& result,
        const std::vector<std::vector<float>>& heatmapData2D,
        const std::vector<std::vector<cv::Point2f>>& contours);

    // 从热力图数据创建图像
    cv::Mat createHeatmapImageFromData(const std::vector<std::vector<float>>& heatmapData2D,
        bool denoise = true,
        float threshold = 0.03f);

    // 从热力图数据提取轮廓点集
    std::vector<std::vector<cv::Point2f>> extractContours(
        const std::vector<std::vector<float>>& heatmapData2D,
        int threshold = 10);

    // 获取网络信息
    int getNetWidth() const { return m_netWidth; }
    int getNetHeight() const { return m_netHeight; }

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
    bool initialize(const std::string& paramPath, const std::string& binPath, int numThreads);
    bool initializeCls(const std::string& paramPath, const std::string& binPath, int numThreads);

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

    cv::Mat applyHeatmapColormap(const cv::Mat& heatmap);

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

    // 在热力图上绘制数值网格（数值背景图层）
    void drawValueGrid(cv::Mat& img,
        const std::vector<std::vector<float>>& heatmapData2D,
        int scale = 20);

    // 绘制火柴人骨架（公共逻辑）
    void drawSkeleton(cv::Mat& drawImg,
        const std::vector<HeatmapResult>& result,
        std::function<cv::Point2f(float, float)> convertCoords,
        int lineScale = 2,
        int pointRadius = 4);

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

    // NCNN
    ncnn::Net m_net;

    // 输入输出信息
    std::string m_inputName;
    std::string m_outputName;
    std::vector<int> m_outputShape;

    // 推理输出
    ncnn::Mat m_outputMat;

    // 颜色映射表
    cv::Mat m_colormapTable;

    // 当前热力图列数（32 或 64）
    int m_heatmapCols;
};
