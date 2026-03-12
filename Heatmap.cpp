#include "ObjectDetectInference.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#ifndef __EMSCRIPTEN__
#include <omp.h>
#endif
#include <cstring>

//初始化精确颜色映射表
void YoloNcnn::initColormapTableExact() {
    m_colormapTable = cv::Mat(256, 1, CV_8UC3);

    std::vector<cv::Vec3b> keyColors = {
        cv::Vec3b(255, 255, 255),   // 白色
        cv::Vec3b(127, 0, 0),       // 深蓝色
        cv::Vec3b(204, 51, 0),      // 蓝色
        cv::Vec3b(229, 153, 0),     // 青色
        cv::Vec3b(153, 229, 0),     // 蓝绿色
        cv::Vec3b(0, 255, 127),     // 黄绿色
        cv::Vec3b(0, 255, 255),     // 黄色
        cv::Vec3b(0, 204, 255)      // 橙黄色
    };

    uchar* colormapData = m_colormapTable.ptr<uchar>(0);

    // 预先计算所有的alpha值
    std::vector<float> alphas(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f * 7.0f;
        alphas[i] = t - std::floor(t);
    }

    // 使用指针和预计算的alphas进行颜色插值
    for (int i = 0; i < 256; ++i) {
        int idx1 = static_cast<int>(i / 255.0f * 7.0f);
        int idx2 = std::min(idx1 + 1, 7);
        float alpha = alphas[i];

        const cv::Vec3b& color1 = keyColors[idx1];
        const cv::Vec3b& color2 = keyColors[idx2];

        for (int c = 0; c < 3; ++c) {
            colormapData[i * 3 + c] = static_cast<uchar>(
                color1[c] * (1 - alpha) + color2[c] * alpha);
        }
    }

    // 确保索引0对应白色
    m_colormapTable.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);
}


cv::Mat YoloNcnn::applyHeatmapColormap(const cv::Mat& heatmap) {
    if (heatmap.empty()) {
        return cv::Mat();
    }

    cv::Mat colorImage;

    // 使用自定义颜色映射表
    cv::applyColorMap(heatmap, colorImage, m_colormapTable);

    return colorImage;
}

// 公共热力图预处理：数据转换 + 去噪
cv::Mat YoloNcnn::prepareHeatmap(const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold) {

    if (heatmapData2D.empty() || heatmapData2D[0].empty()) {
        return cv::Mat();
    }

    int rows = static_cast<int>(heatmapData2D.size());
    int cols = static_cast<int>(heatmapData2D[0].size());

    cv::Mat heatmap(rows, cols, CV_32FC1);
#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        const auto& row = heatmapData2D[i];
        float* dst = heatmap.ptr<float>(i);
        memcpy(dst, row.data(), cols * sizeof(float));
    }

    if (denoise) {
        double currentMaxVal;
        cv::minMaxLoc(heatmap, nullptr, &currentMaxVal);
        float thresholdValue = static_cast<float>(threshold * currentMaxVal);
        cv::threshold(heatmap, heatmap, thresholdValue, 0, cv::THRESH_TOZERO);
        cv::medianBlur(heatmap, heatmap, 3);
    }

    return heatmap;
}

cv::Mat YoloNcnn::processHeatmapData(const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold, int scale) {

    // 使用公共预处理函数
    cv::Mat heatmap = prepareHeatmap(heatmapData2D, denoise, threshold);
    if (heatmap.empty()) {
        return cv::Mat();
    }

    int rows = heatmap.rows;
    int cols = heatmap.cols;

    // 归一化处理（用于颜色映射）
    cv::Mat normalizedForColormap;
    double maxVal;
    cv::minMaxLoc(heatmap, nullptr, &maxVal);

    if (maxVal > 0) {
        normalizedForColormap = heatmap / maxVal;
    }
    else {
        normalizedForColormap = cv::Mat::zeros(rows, cols, CV_32F);
    }

    // 转换为8位图像（0-255），用于颜色映射
    cv::Mat heatmap8U;
    normalizedForColormap.convertTo(heatmap8U, CV_8UC1, 255.0);

    // 应用颜色映射
    cv::Mat colorImage = applyHeatmapColormap(heatmap8U);

    // 将0值设为白色
    cv::Mat mask = (heatmap8U == 0);
    colorImage.setTo(cv::Vec3b(255, 255, 255), mask);

    // 放大图像
    cv::Mat enlargedImage;
    cv::resize(colorImage, enlargedImage, cv::Size(cols * scale, rows * scale),
        0, 0, cv::INTER_NEAREST);
    return enlargedImage;
}

cv::Mat YoloNcnn::createHeatmapImageFromData(const std::vector<std::vector<float>>& heatmapData2D,
    bool denoise, float threshold) {
    return processHeatmapData(heatmapData2D, denoise, threshold);
}

std::vector<std::vector<cv::Point2f>> YoloNcnn::extractContours(
    const std::vector<std::vector<float>>& heatmapData2D,
    int threshold) {
    
    std::vector<std::vector<cv::Point2f>> result;
    
    // 使用公共预处理函数（去噪参数与 processHeatmapData 一致）
    cv::Mat heatmap = prepareHeatmap(heatmapData2D, true, 0.03f);
    if (heatmap.empty()) {
        return result;
    }
    
    const int rows = heatmap.rows;
    const int cols = heatmap.cols;
    const int scale = 10;  // 放大倍数
    
    // 归一化到 0-255 并转换为 CV_8U
    double maxVal;
    cv::minMaxLoc(heatmap, nullptr, &maxVal);
    
    cv::Mat gray;
    if (maxVal > 0) {
        heatmap.convertTo(gray, CV_8UC1, 255.0 / maxVal);
    } else {
        return result;  // 全零热力图，无轮廓
    }
    
    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);
    
    // 放大10倍
    cv::Mat binaryLarge;
    cv::resize(binary, binaryLarge, cv::Size(cols * scale, rows * scale), 
               0, 0, cv::INTER_NEAREST);

    //在放大后的图像上检测外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryLarge, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    //将轮廓坐标缩小10倍，转换回原始坐标系
    result.reserve(contours.size());
    for (const auto& contour : contours) {
        std::vector<cv::Point2f> origContour;
        origContour.reserve(contour.size());
        for (const auto& pt : contour) {
            origContour.emplace_back(
                pt.x / static_cast<float>(scale),
                pt.y / static_cast<float>(scale)
            );
        }
        result.push_back(std::move(origContour));
    }
    
    return result;
}

void YoloNcnn::drawPredOnHeatmap(cv::Mat& img, const std::vector<HeatmapResult>& result) {
    if (img.empty()) {
        std::cerr << "输入图像为空!" << std::endl;
        return;
    }

    cv::Mat drawImg = img.clone();
    cv::Mat overlay = drawImg.clone();
    int boxThickness = 1;
    float alpha = 0.5f;
    float borderDarkenFactor = 0.6f;
    
    // 转换坐标的辅助函数
    auto convertToImageCoords = [](float cx, float cy) -> cv::Point2f {
        return cv::Point2f(cx * 10.0f, cy * 10.0f);
    };

    // 颜色映射
    std::map<int, cv::Scalar> colorMap = {
        {0, cv::Scalar(255, 255, 0)},   // 手臂
        {1, cv::Scalar(0, 165, 255)},   // 胸
        {2, cv::Scalar(255, 0, 255)},   // 臀
        {3, cv::Scalar(0, 255, 255)},   // 头
        {4, cv::Scalar(255, 255, 255)}, // human
        {5, cv::Scalar(226, 43, 138)},  // 腿
        {6, cv::Scalar(0, 0, 255)},     // 肩膀
        {7, cv::Scalar(180, 105, 255)}  // 腰
    };

    //绘制检测框和标签
    for (const auto& r : result) {
        float scale_x = 640.0f / 64.0f;
        float scale_y = 320.0f / 32.0f;
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Scalar boxColor = (colorMap.find(r.id) != colorMap.end()) 
                ? colorMap[r.id] : cv::Scalar(128, 128, 128);
            cv::Scalar borderColor = boxColor * borderDarkenFactor;

            // 填充检测框（ID4 human 不填充）
            if (r.id != 4) {
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                cv::Point pts[4];
                for (int i = 0; i < 4; ++i) {
                    pts[i] = cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));
                }
                cv::fillConvexPoly(overlay, pts, 4, boxColor);
            }

            // 绘制边框
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int l = 0; l < 4; ++l) {
                cv::line(drawImg, vertices[l], vertices[(l + 1) % 4], borderColor, boxThickness, 8);
            }

            // 绘制ID和置信度标签
            std::ostringstream label;
            label << r.id << ":" << std::fixed << std::setprecision(2) << r.confidence;
            
            // 找旋转框左上角顶点
            int topLeftIdx = 0;
            float minY = vertices[0].y;
            for (int i = 1; i < 4; ++i) {
                if (vertices[i].y < minY || (vertices[i].y == minY && vertices[i].x < vertices[topLeftIdx].x)) {
                    minY = vertices[i].y;
                    topLeftIdx = i;
                }
            }
            
            int baseline = 0;
            double fontScale = 0.35;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            
            int textX = static_cast<int>(vertices[topLeftIdx].x) + 2;
            int textY = static_cast<int>(vertices[topLeftIdx].y) + textSize.height + 2;
            
            textX = std::max(1, std::min(textX, drawImg.cols - textSize.width - 2));
            textY = std::max(textSize.height + 1, std::min(textY, drawImg.rows - 2));
            
            cv::rectangle(drawImg,
                cv::Point(textX - 1, textY - textSize.height - 1),
                cv::Point(textX + textSize.width + 1, textY + baseline + 1),
                cv::Scalar(255, 255, 255), -1);
            
            cv::putText(drawImg, label.str(), cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
        }
    }
    
    // 混合半透明填充层
    cv::addWeighted(overlay, alpha, drawImg, 1 - alpha, 0, drawImg);

    //绘制火柴人骨架 
    drawSkeleton(drawImg, result, convertToImageCoords, 2, 4);

    drawImg.copyTo(img);
}

//带轮廓交集的绘制
void YoloNcnn::drawPredOnHeatmap(cv::Mat& img, const std::vector<HeatmapResult>& result,
    const std::vector<std::vector<cv::Point2f>>& contours) {
    
    if (img.empty()) {
        std::cerr << "输入图像为空!" << std::endl;
        return;
    }

    cv::Mat drawImg = img.clone();
    cv::Mat overlay = drawImg.clone();
    float alpha = 0.8f;  // 不透明度 0.8
    cv::Scalar borderColor(0, 0, 0);  // 统一黑色边框
    int boxThickness = 1;
    
    // 坐标转换
    auto convertToImageCoords = [](float cx, float cy) -> cv::Point2f {
        return cv::Point2f(cx * 10.0f, cy * 10.0f);
    };

    // 颜色映射
    std::map<int, cv::Scalar> colorMap = {
        {0, cv::Scalar(255, 255, 0)},   // 手臂
        {1, cv::Scalar(0, 165, 255)},   // 胸
        {2, cv::Scalar(255, 0, 255)},   // 臀
        {3, cv::Scalar(0, 255, 255)},   // 头
        {4, cv::Scalar(255, 255, 255)}, // human
        {5, cv::Scalar(226, 43, 138)},  // 腿
        {6, cv::Scalar(0, 0, 255)},     // 肩膀
        {7, cv::Scalar(180, 105, 255)}  // 腰
    };

    // 将所有轮廓合并为一个大的轮廓多边形集合
    std::vector<std::vector<cv::Point>> scaledContours;
    for (const auto& contour : contours) {
        std::vector<cv::Point> scaled;
        for (const auto& pt : contour) {
            scaled.emplace_back(static_cast<int>(pt.x * 10), static_cast<int>(pt.y * 10));
        }
        if (!scaled.empty()) {
            scaledContours.push_back(scaled);
        }
    }

    // 预先创建轮廓掩码
    cv::Mat contourMask;
    if (!scaledContours.empty()) {
        contourMask = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::drawContours(contourMask, scaledContours, -1, cv::Scalar(255), cv::FILLED);
    }

    //绘制检测框交集区域和边框标签
    for (const auto& r : result) {
        float scale_x = 640.0f / 64.0f;
        float scale_y = 320.0f / 32.0f;
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Scalar boxColor = (colorMap.find(r.id) != colorMap.end()) 
                ? colorMap[r.id] : cv::Scalar(128, 128, 128);

            // 获取 OBB 四个顶点
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            
            // 将 OBB 转换为多边形
            std::vector<cv::Point> obbPoly;
            for (int i = 0; i < 4; ++i) {
                obbPoly.emplace_back(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));
            }

            // 计算 OBB 与轮廓的交集并填充（ID4 human 不填充）
            if (r.id != 4 && !contourMask.empty()) {
                // 创建 OBB 掩码
                cv::Mat obbMask = cv::Mat::zeros(img.size(), CV_8UC1);
                cv::fillConvexPoly(obbMask, obbPoly, cv::Scalar(255));
                
                // 计算交集掩码
                cv::Mat intersectionMask;
                cv::bitwise_and(obbMask, contourMask, intersectionMask);
                
                // 在 overlay 上填充交集区域
                overlay.setTo(boxColor, intersectionMask);
            }
        }
    }
    
    // 混合半透明交集填充层（alpha = 0.8）
    cv::addWeighted(overlay, alpha, drawImg, 1 - alpha, 0, drawImg);

    // 混合后绘制边框和标签
    for (const auto& r : result) {
        float scale_x = 640.0f / 64.0f;
        float scale_y = 320.0f / 32.0f;
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            
            // 绘制黑色边框
            for (int l = 0; l < 4; ++l) {
                cv::line(drawImg, vertices[l], vertices[(l + 1) % 4], borderColor, boxThickness, 8);
            }

            // 绘制 ID 和置信度标签
            std::ostringstream label;
            label << r.id << ":" << std::fixed << std::setprecision(2) << r.confidence;
            
            int topLeftIdx = 0;
            float minY = vertices[0].y;
            for (int i = 1; i < 4; ++i) {
                if (vertices[i].y < minY || (vertices[i].y == minY && vertices[i].x < vertices[topLeftIdx].x)) {
                    minY = vertices[i].y;
                    topLeftIdx = i;
                }
            }
            
            int baseline = 0;
            double fontScale = 0.35;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            
            int textX = static_cast<int>(vertices[topLeftIdx].x) + 2;
            int textY = static_cast<int>(vertices[topLeftIdx].y) + textSize.height + 2;
            
            textX = std::max(1, std::min(textX, drawImg.cols - textSize.width - 2));
            textY = std::max(textSize.height + 1, std::min(textY, drawImg.rows - 2));
            
            cv::rectangle(drawImg,
                cv::Point(textX - 1, textY - textSize.height - 1),
                cv::Point(textX + textSize.width + 1, textY + baseline + 1),
                cv::Scalar(255, 255, 255), -1);
            
            cv::putText(drawImg, label.str(), cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness);
        }
    }

    // 绘制火柴人骨架
    drawSkeleton(drawImg, result, convertToImageCoords, 2, 4);

    drawImg.copyTo(img);
}

// 数值网格背景绘制
void YoloNcnn::drawValueGrid(cv::Mat& img, const std::vector<std::vector<float>>& heatmapData2D, int scale) {
    if (img.empty() || heatmapData2D.empty()) return;

    int rows = static_cast<int>(heatmapData2D.size());
    int cols = static_cast<int>(heatmapData2D[0].size());

    // 绘制网格线（浅灰色）
    cv::Scalar gridColor(200, 200, 200);
    for (int r = 0; r <= rows; ++r) {
        cv::line(img, cv::Point(0, r * scale), cv::Point(cols * scale, r * scale), gridColor, 1);
    }
    for (int c = 0; c <= cols; ++c) {
        cv::line(img, cv::Point(c * scale, 0), cv::Point(c * scale, rows * scale), gridColor, 1);
    }

    // 在每个格子中央绘制数值
    double fontScale = 0.3;
    int thickness = 1;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float val = heatmapData2D[r][c];
            int intVal = static_cast<int>(std::round(val));
            std::string text = std::to_string(intVal);

            // 计算格子中心位置
            int cellCenterX = c * scale + scale / 2;
            int cellCenterY = r * scale + scale / 2;

            // 根据背景亮度选择文字颜色，零值用更浅的颜色
            cv::Vec3b bgPixel = img.at<cv::Vec3b>(cellCenterY, cellCenterX);
            float brightness = 0.299f * bgPixel[2] + 0.587f * bgPixel[1] + 0.114f * bgPixel[0];
            cv::Scalar textColor;
            if (intVal == 0) {
                textColor = (brightness > 128) ? cv::Scalar(180, 180, 180) : cv::Scalar(100, 100, 100);
            } else {
                textColor = (brightness > 128) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
            }

            // 居中绘制文字
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            int textX = cellCenterX - textSize.width / 2;
            int textY = cellCenterY + textSize.height / 2;
            cv::putText(img, text, cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
        }
    }
}

//火柴人骨架绘制
void YoloNcnn::drawSkeleton(cv::Mat& drawImg,
    const std::vector<HeatmapResult>& result,
    std::function<cv::Point2f(float, float)> convertCoords,
    int lineScale, int pointRadius) {

    std::map<int, std::vector<HeatmapResult>> grouped;
    for (const auto& r : result) {
        if (r.id == 4) continue;  // 跳过human
        grouped[r.id].push_back(r);
    }

    cv::Scalar connectionColor(147, 20, 255);  // 粉红色连线
    cv::Scalar pointColor(0, 0, 0);             // 黑色中心点

    // 收集主干点：头(3) -> 胸(1) -> 腰(7) -> 臀(2)
    std::map<int, cv::Point2f> trunkPoints;
    std::vector<int> trunkIDs = { 3, 1, 7, 2 };
    for (int id : trunkIDs) {
        if (grouped.find(id) != grouped.end() && !grouped[id].empty()) {
            const auto& r = grouped[id][0];
            trunkPoints[id] = convertCoords(r.cx, r.cy);
        }
    }

    // 连接主干点
    for (size_t i = 0; i < trunkIDs.size() - 1; ++i) {
        int id1 = trunkIDs[i];
        int id2 = trunkIDs[i + 1];
        if (trunkPoints.find(id1) != trunkPoints.end() &&
            trunkPoints.find(id2) != trunkPoints.end()) {
            cv::line(drawImg, trunkPoints[id1], trunkPoints[id2], connectionColor, lineScale);
        }
    }

    // 肩膀处理
    if (grouped.find(6) != grouped.end() && !grouped[6].empty()) {
        std::vector<HeatmapResult> shoulders = grouped[6];
        std::sort(shoulders.begin(), shoulders.end(),
            [](const HeatmapResult& a, const HeatmapResult& b) { return a.cx < b.cx; });

        if (shoulders.size() == 2) {
            cv::Point2f leftShoulder = convertCoords(shoulders[0].cx, shoulders[0].cy);
            cv::Point2f rightShoulder = convertCoords(shoulders[1].cx, shoulders[1].cy);
            cv::line(drawImg, leftShoulder, rightShoulder, connectionColor, lineScale);
        }

        if (trunkPoints.find(1) != trunkPoints.end()) {
            for (const auto& shoulder : shoulders) {
                cv::Point2f shoulderPoint = convertCoords(shoulder.cx, shoulder.cy);
                cv::line(drawImg, trunkPoints[1], shoulderPoint, connectionColor, lineScale);
            }
        }
    }

    // 手臂连接
    if (grouped.find(0) != grouped.end() && !grouped[0].empty()) {
        std::vector<HeatmapResult> arms = grouped[0];
        bool hasShoulders = (grouped.find(6) != grouped.end() && !grouped[6].empty());

        if (hasShoulders) {
            std::vector<HeatmapResult> shoulders = grouped[6];
            for (const auto& arm : arms) {
                cv::Point2f armPoint = convertCoords(arm.cx, arm.cy);
                cv::Point2f closestShoulder(0, 0);
                float minDistance = std::numeric_limits<float>::max();
                for (const auto& shoulder : shoulders) {
                    cv::Point2f shoulderPoint = convertCoords(shoulder.cx, shoulder.cy);
                    float dx = shoulderPoint.x - armPoint.x;
                    float dy = shoulderPoint.y - armPoint.y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestShoulder = shoulderPoint;
                    }
                }
                cv::line(drawImg, armPoint, closestShoulder, connectionColor, lineScale);
            }
        } else if (trunkPoints.find(1) != trunkPoints.end()) {
            for (const auto& arm : arms) {
                cv::Point2f armPoint = convertCoords(arm.cx, arm.cy);
                cv::line(drawImg, armPoint, trunkPoints[1], connectionColor, lineScale);
            }
        }
    }

    // 腿部连接
    if (grouped.find(5) != grouped.end() && !grouped[5].empty()) {
        std::vector<HeatmapResult> legs = grouped[5];

        if (trunkPoints.find(2) != trunkPoints.end()) {
            for (const auto& leg : legs) {
                cv::Point2f legPoint = convertCoords(leg.cx, leg.cy);
                cv::line(drawImg, legPoint, trunkPoints[2], connectionColor, lineScale);
            }
        } else if (trunkPoints.find(7) != trunkPoints.end()) {
            for (const auto& leg : legs) {
                cv::Point2f legPoint = convertCoords(leg.cx, leg.cy);
                cv::line(drawImg, legPoint, trunkPoints[7], connectionColor, lineScale);
            }
        }
    }

    // 绘制所有中心点
    for (const auto& r : result) {
        if (r.id == 4) continue;
        cv::Point2f center = convertCoords(r.cx, r.cy);
        cv::circle(drawImg, center, pointRadius, pointColor, -1);
    }
}

// 带数值网格背景的绘制（无轮廓）
void YoloNcnn::drawPredOnHeatmap(cv::Mat& img, const std::vector<HeatmapResult>& result,
    const std::vector<std::vector<float>>& heatmapData2D) {

    if (img.empty()) return;

    // 放大图像到 1280×640（scale=20）
    int rows = static_cast<int>(heatmapData2D.size());     // 32
    int cols = static_cast<int>(heatmapData2D[0].size());   // 64
    int scale = 20;
    cv::resize(img, img, cv::Size(cols * scale, rows * scale), 0, 0, cv::INTER_NEAREST);

    // 先绘制数值网格背景
    drawValueGrid(img, heatmapData2D, scale);
    
    cv::Mat drawImg = img.clone();
    cv::Mat overlay = drawImg.clone();
    int boxThickness = 2;
    float alpha = 0.5f;
    float borderDarkenFactor = 0.6f;

    auto convertCoords = [scale](float cx, float cy) -> cv::Point2f {
        return cv::Point2f(cx * static_cast<float>(scale), cy * static_cast<float>(scale));
    };

    std::map<int, cv::Scalar> colorMap = {
        {0, cv::Scalar(255, 255, 0)},   // 手臂
        {1, cv::Scalar(0, 165, 255)},   // 胸
        {2, cv::Scalar(255, 0, 255)},   // 臀
        {3, cv::Scalar(0, 255, 255)},   // 头
        {4, cv::Scalar(255, 255, 255)}, // human
        {5, cv::Scalar(226, 43, 138)},  // 腿
        {6, cv::Scalar(0, 0, 255)},     // 肩膀
        {7, cv::Scalar(180, 105, 255)}  // 腰
    };

    float scale_x = static_cast<float>(cols * scale) / static_cast<float>(cols);  // = scale
    float scale_y = static_cast<float>(rows * scale) / static_cast<float>(rows);  // = scale

    // 1. 绘制检测框和标签
    for (const auto& r : result) {
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Scalar boxColor = (colorMap.find(r.id) != colorMap.end())
                ? colorMap[r.id] : cv::Scalar(128, 128, 128);
            cv::Scalar borderClr = boxColor * borderDarkenFactor;

            if (r.id != 4) {
                cv::Point2f vertices[4];
                rotatedRect.points(vertices);
                cv::Point pts[4];
                for (int i = 0; i < 4; ++i) {
                    pts[i] = cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));
                }
                cv::fillConvexPoly(overlay, pts, 4, boxColor);
            }

            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int l = 0; l < 4; ++l) {
                cv::line(drawImg, vertices[l], vertices[(l + 1) % 4], borderClr, boxThickness, 8);
            }

            std::ostringstream label;
            label << r.id << ":" << std::fixed << std::setprecision(2) << r.confidence;

            int topLeftIdx = 0;
            float minY = vertices[0].y;
            for (int i = 1; i < 4; ++i) {
                if (vertices[i].y < minY || (vertices[i].y == minY && vertices[i].x < vertices[topLeftIdx].x)) {
                    minY = vertices[i].y;
                    topLeftIdx = i;
                }
            }

            int baseline = 0;
            double fScale = 0.45;
            int thick = 1;
            cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, fScale, thick, &baseline);

            int textX = static_cast<int>(vertices[topLeftIdx].x) + 2;
            int textY = static_cast<int>(vertices[topLeftIdx].y) + textSize.height + 2;
            textX = std::max(1, std::min(textX, drawImg.cols - textSize.width - 2));
            textY = std::max(textSize.height + 1, std::min(textY, drawImg.rows - 2));

            cv::rectangle(drawImg,
                cv::Point(textX - 1, textY - textSize.height - 1),
                cv::Point(textX + textSize.width + 1, textY + baseline + 1),
                cv::Scalar(255, 255, 255), -1);
            cv::putText(drawImg, label.str(), cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fScale, cv::Scalar(0, 0, 0), thick);
        }
    }

    cv::addWeighted(overlay, alpha, drawImg, 1 - alpha, 0, drawImg);

    // 2. 绘制火柴人骨架
    drawSkeleton(drawImg, result, convertCoords, 3, 5);

    drawImg.copyTo(img);
}

// 带数值网格背景 + 轮廓交集的绘制
void YoloNcnn::drawPredOnHeatmap(cv::Mat& img, const std::vector<HeatmapResult>& result,
    const std::vector<std::vector<float>>& heatmapData2D,
    const std::vector<std::vector<cv::Point2f>>& contours) {

    if (img.empty()) return;

    int rows = static_cast<int>(heatmapData2D.size());     // 32
    int cols = static_cast<int>(heatmapData2D[0].size());   // 64
    int scale = 20;
    cv::resize(img, img, cv::Size(cols * scale, rows * scale), 0, 0, cv::INTER_NEAREST);

    // 先绘制数值网格背景
    drawValueGrid(img, heatmapData2D, scale);

    cv::Mat drawImg = img.clone();
    cv::Mat overlay = drawImg.clone();
    float alpha = 0.8f;
    cv::Scalar borderColor(0, 0, 0);
    int boxThickness = 2;

    auto convertCoords = [scale](float cx, float cy) -> cv::Point2f {
        return cv::Point2f(cx * static_cast<float>(scale), cy * static_cast<float>(scale));
    };

    std::map<int, cv::Scalar> colorMap = {
        {0, cv::Scalar(255, 255, 0)},
        {1, cv::Scalar(0, 165, 255)},
        {2, cv::Scalar(255, 0, 255)},
        {3, cv::Scalar(0, 255, 255)},
        {4, cv::Scalar(255, 255, 255)},
        {5, cv::Scalar(226, 43, 138)},
        {6, cv::Scalar(0, 0, 255)},
        {7, cv::Scalar(180, 105, 255)}
    };

    auto getPartName = [](int id) -> const char* {
        static const char* partNames[] = {
            "arm", "brust", "buttocks", "head",
            "human", "leg", "shoulder", "waist"
        };
        if (id >= 0 && id <= 7) {
            return partNames[id];
        }
        return "unknown";
    };

    // 缩放轮廓到图像坐标
    std::vector<std::vector<cv::Point>> scaledContours;
    for (const auto& contour : contours) {
        std::vector<cv::Point> scaled;
        for (const auto& pt : contour) {
            scaled.emplace_back(static_cast<int>(pt.x * scale), static_cast<int>(pt.y * scale));
        }
        if (!scaled.empty()) scaledContours.push_back(scaled);
    }

    float scale_x = static_cast<float>(scale);
    float scale_y = static_cast<float>(scale);

    // 预先创建轮廓掩码
    cv::Mat contourMask;
    if (!scaledContours.empty()) {
        contourMask = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::drawContours(contourMask, scaledContours, -1, cv::Scalar(255), cv::FILLED);
    }

    // 绘制检测框交集区域
    for (const auto& r : result) {
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Scalar boxColor = (colorMap.find(r.id) != colorMap.end())
                ? colorMap[r.id] : cv::Scalar(128, 128, 128);

            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            std::vector<cv::Point> obbPoly;
            for (int i = 0; i < 4; ++i) {
                obbPoly.emplace_back(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));
            }

            if (r.id != 4 && !contourMask.empty()) {
                cv::Mat obbMask = cv::Mat::zeros(img.size(), CV_8UC1);
                cv::fillConvexPoly(obbMask, obbPoly, cv::Scalar(255));
                cv::Mat intersectionMask;
                cv::bitwise_and(obbMask, contourMask, intersectionMask);
                overlay.setTo(boxColor, intersectionMask);
            }
        }
    }

    cv::addWeighted(overlay, alpha, drawImg, 1 - alpha, 0, drawImg);

    // 绘制边框和标签
    for (const auto& r : result) {
        cv::Point2f center(r.cx * scale_x, r.cy * scale_y);
        cv::Size2f size(r.l * scale_x, r.s * scale_y);
        float angle = r.angle * 180.0f / CV_PI;
        cv::RotatedRect rotatedRect(center, size, angle);

        if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);
            for (int l = 0; l < 4; ++l) {
                cv::line(drawImg, vertices[l], vertices[(l + 1) % 4], borderColor, boxThickness, 8);
            }

            std::string label = getPartName(r.id);
            int baseline = 0;
            double fScale = 0.60;  
            int thick = 2;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fScale, thick, &baseline);

            // 控制标签尺寸，尽量保持在旋转框内部
            const float maxTextW = std::max(8.0f, rotatedRect.size.width - 6.0f);
            const float maxTextH = std::max(8.0f, rotatedRect.size.height - 6.0f);
            while ((textSize.width > maxTextW || (textSize.height + baseline) > maxTextH) && fScale > 0.35) {
                fScale -= 0.05;
                thick = (fScale >= 0.55) ? 2 : 1;
                textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fScale, thick, &baseline);
            }

            // 标签放在旋转框内左上角（由左上顶点向中心内缩，避免压线）
            int topLeftIdx = 0;
            float minY = vertices[0].y;
            for (int i = 1; i < 4; ++i) {
                if (vertices[i].y < minY || (vertices[i].y == minY && vertices[i].x < vertices[topLeftIdx].x)) {
                    minY = vertices[i].y;
                    topLeftIdx = i;
                }
            }

            cv::Point2f topLeftVertex = vertices[topLeftIdx];
            cv::Point2f anchor = topLeftVertex + 0.15f * (center - topLeftVertex);

            int textX = static_cast<int>(anchor.x) + 2;
            int textY = static_cast<int>(anchor.y) + textSize.height + 2;

            // 防止越界
            textX = std::max(0, std::min(textX, drawImg.cols - textSize.width - 1));
            textY = std::max(textSize.height + 1, std::min(textY, drawImg.rows - baseline - 2));

            cv::rectangle(drawImg,
                cv::Point(textX - 3, textY - textSize.height - 2),
                cv::Point(textX + textSize.width + 3, textY + baseline + 2),
                cv::Scalar(0, 0, 0), -1);  // 黑底
            cv::putText(drawImg, label, cv::Point(textX, textY),
                cv::FONT_HERSHEY_SIMPLEX, fScale, cv::Scalar(255, 255, 255), thick);  // 白字
        }
    }

    //绘制火柴人骨架
    drawSkeleton(drawImg, result, convertCoords, 3, 5);

    drawImg.copyTo(img);
}
