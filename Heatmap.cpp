#include "ObjectDetectInference.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifndef __EMSCRIPTEN__
#include <omp.h>
#endif

// 初始化 256 级颜色映射表（0 值固定为白色）
void YoloNcnn::initColormapTableExact() {
    m_colormapTable = cv::Mat(256, 1, CV_8UC3);

    const std::vector<cv::Vec3b> keyColors = {
        cv::Vec3b(255, 255, 255),
        cv::Vec3b(127, 0, 0),
        cv::Vec3b(204, 51, 0),
        cv::Vec3b(229, 153, 0),
        cv::Vec3b(153, 229, 0),
        cv::Vec3b(0, 255, 127),
        cv::Vec3b(0, 255, 255),
        cv::Vec3b(0, 204, 255)
    };

    uchar* colormapData = m_colormapTable.ptr<uchar>(0);
    for (int i = 0; i < 256; ++i) {
        const float t = i / 255.0f * 7.0f;
        const int idx1 = static_cast<int>(t);
        const int idx2 = std::min(idx1 + 1, 7);
        const float alpha = t - std::floor(t);
        for (int c = 0; c < 3; ++c) {
            colormapData[i * 3 + c] = static_cast<uchar>(
                keyColors[idx1][c] * (1.0f - alpha) + keyColors[idx2][c] * alpha
            );
        }
    }
    m_colormapTable.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);
}

cv::Mat YoloNcnn::prepareHeatmap(const std::vector<std::vector<float>>& heatmapData2D,
                                 bool denoise,
                                 float threshold) {
    if (heatmapData2D.empty() || heatmapData2D[0].empty()) {
        return cv::Mat();
    }

    const int rows = static_cast<int>(heatmapData2D.size());
    const int cols = static_cast<int>(heatmapData2D[0].size());
    // 复用 Mat 缓冲，减少每帧分配
    m_heatmapFloatBuffer.create(rows, cols, CV_32FC1);

    // 固定输入尺寸（32x64）走快路径，避免额外分支开销
    if (rows == 32 && cols == 64) {
        for (int i = 0; i < rows; ++i) {
            const auto& row = heatmapData2D[i];
            if (static_cast<int>(row.size()) < cols) {
                return cv::Mat();
            }
            std::memcpy(m_heatmapFloatBuffer.ptr<float>(i), row.data(), static_cast<size_t>(cols) * sizeof(float));
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            const auto& row = heatmapData2D[i];
            if (static_cast<int>(row.size()) < cols) {
                continue;
            }
            std::memcpy(m_heatmapFloatBuffer.ptr<float>(i), row.data(), static_cast<size_t>(cols) * sizeof(float));
        }
    }

    if (denoise) {
        // 去噪：按当前帧最大值做阈值截断，再做中值滤波
        double currentMaxVal = 0.0;
        cv::minMaxLoc(m_heatmapFloatBuffer, nullptr, &currentMaxVal);
        const float thresholdValue = static_cast<float>(threshold * currentMaxVal);
        cv::threshold(m_heatmapFloatBuffer, m_heatmapFloatBuffer, thresholdValue, 0, cv::THRESH_TOZERO);
        cv::medianBlur(m_heatmapFloatBuffer, m_heatmapFloatBuffer, 3);
    }

    return m_heatmapFloatBuffer;
}

cv::Mat YoloNcnn::prepareHeatmap(const float* flatData,
                                 int rows,
                                 int cols,
                                 bool denoise,
                                 float threshold) {
    if (!flatData || rows <= 0 || cols <= 0) {
        return cv::Mat();
    }

    // 扁平输入直接拷贝到连续 Mat
    m_heatmapFloatBuffer.create(rows, cols, CV_32FC1);
    std::memcpy(
        m_heatmapFloatBuffer.data,
        flatData,
        static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float)
    );

    if (denoise) {
        // 去噪：按比例阈值抑制低响应，再做中值滤波
        double currentMaxVal = 0.0;
        cv::minMaxLoc(m_heatmapFloatBuffer, nullptr, &currentMaxVal);
        const float thresholdValue = static_cast<float>(threshold * currentMaxVal);
        cv::threshold(m_heatmapFloatBuffer, m_heatmapFloatBuffer, thresholdValue, 0, cv::THRESH_TOZERO);
        cv::medianBlur(m_heatmapFloatBuffer, m_heatmapFloatBuffer, 3);
    }

    return m_heatmapFloatBuffer;
}

cv::Mat YoloNcnn::processHeatmapData(const std::vector<std::vector<float>>& heatmapData2D,
                                     bool denoise,
                                     float threshold,
                                     int scale) {
    // 预处理：数据转换 + 可选去噪
    cv::Mat heatmap = prepareHeatmap(heatmapData2D, denoise, threshold);
    if (heatmap.empty()) {
        return cv::Mat();
    }

    const int rows = heatmap.rows;
    const int cols = heatmap.cols;

    // 归一化到 [0, 1]，用于后续颜色映射
    m_heatmapNormalizedBuffer.create(rows, cols, CV_32FC1);
    double maxVal = 0.0;
    cv::minMaxLoc(heatmap, nullptr, &maxVal);

    if (maxVal > 0.0) {
        heatmap.convertTo(m_heatmapNormalizedBuffer, CV_32FC1, 1.0 / maxVal);
    } else {
        m_heatmapNormalizedBuffer.setTo(0);
    }

    // 转成 8bit 后套用颜色映射，0 值强制置白
    m_heatmap8UBuffer.create(rows, cols, CV_8UC1);
    m_heatmapNormalizedBuffer.convertTo(m_heatmap8UBuffer, CV_8UC1, 255.0);
    cv::applyColorMap(m_heatmap8UBuffer, m_heatmapColorBuffer, m_colormapTable);
    cv::Mat mask = (m_heatmap8UBuffer == 0);
    m_heatmapColorBuffer.setTo(cv::Vec3b(255, 255, 255), mask);

    // 放大到网络前处理所需比例（默认 OBB=10，Cls=5）
    cv::resize(
        m_heatmapColorBuffer,
        m_heatmapEnlargedBuffer,
        cv::Size(cols * scale, rows * scale),
        0,
        0,
        cv::INTER_NEAREST
    );
    return m_heatmapEnlargedBuffer;
}

cv::Mat YoloNcnn::processHeatmapData(const float* flatData,
                                     int rows,
                                     int cols,
                                     bool denoise,
                                     float threshold,
                                     int scale) {
    // 扁平输入路径：预处理 + 颜色映射
    cv::Mat heatmap = prepareHeatmap(flatData, rows, cols, denoise, threshold);
    if (heatmap.empty()) {
        return cv::Mat();
    }

    m_heatmapNormalizedBuffer.create(rows, cols, CV_32FC1);
    double maxVal = 0.0;
    cv::minMaxLoc(heatmap, nullptr, &maxVal);

    if (maxVal > 0.0) {
        heatmap.convertTo(m_heatmapNormalizedBuffer, CV_32FC1, 1.0 / maxVal);
    } else {
        m_heatmapNormalizedBuffer.setTo(0);
    }

    m_heatmap8UBuffer.create(rows, cols, CV_8UC1);
    m_heatmapNormalizedBuffer.convertTo(m_heatmap8UBuffer, CV_8UC1, 255.0);
    cv::applyColorMap(m_heatmap8UBuffer, m_heatmapColorBuffer, m_colormapTable);
    cv::Mat mask = (m_heatmap8UBuffer == 0);
    m_heatmapColorBuffer.setTo(cv::Vec3b(255, 255, 255), mask);

    cv::resize(
        m_heatmapColorBuffer,
        m_heatmapEnlargedBuffer,
        cv::Size(cols * scale, rows * scale),
        0,
        0,
        cv::INTER_NEAREST
    );
    return m_heatmapEnlargedBuffer;
}
