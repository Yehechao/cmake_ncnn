#include "ObjectDetectInference.h"
#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace {
    struct CachedRotatedRect {
        cv::RotatedRect rect;
        cv::Rect2f bounds;
    };

    inline bool boundsOverlap(const cv::Rect2f& a, const cv::Rect2f& b) {
        return a.x < b.x + b.width &&
            b.x < a.x + a.width &&
            a.y < b.y + b.height &&
            b.y < a.y + a.height;
    }

    inline CachedRotatedRect makeCachedRect(const HeatmapResult& result) {
        CachedRotatedRect cached;
        cached.rect = cv::RotatedRect(
            cv::Point2f(result.cx, result.cy),
            cv::Size2f(result.l, result.s),
            result.angle * 180.0f / CV_PI
        );
        cached.bounds = cached.rect.boundingRect2f();
        return cached;
    }

    inline float normalizeAngleToRad(float angleValue) {
        float angleRad = angleValue;
        const float twoPi = 2.0f * static_cast<float>(CV_PI);
        const float oneEightyRad = static_cast<float>(CV_PI);

        // 兼容角度/弧度两种导出格式
        if (std::fabs(angleRad) > twoPi + 1e-3f) {
            angleRad = angleRad * static_cast<float>(CV_PI) / 180.0f;
        }

        while (angleRad < 0.0f) angleRad += oneEightyRad;
        while (angleRad >= oneEightyRad) angleRad -= oneEightyRad;
        return angleRad;
    }
}

// 计算两个旋转框的IoU
float YoloNcnn::calc_rotate_iou(const cv::RotatedRect& rect1,
    const cv::RotatedRect& rect2) {
    const cv::Rect2f bounds1 = rect1.boundingRect2f();
    const cv::Rect2f bounds2 = rect2.boundingRect2f();
    if (!boundsOverlap(bounds1, bounds2)) {
        return 0.0f;
    }

    // 获取旋转框的四个顶点
    cv::Point2f vertices1[4];
    cv::Point2f vertices2[4];
    rect1.points(vertices1);
    rect2.points(vertices2);

    // 计算两个多边形的交集面积
    std::vector<cv::Point2f> intersection;
    cv::rotatedRectangleIntersection(rect1, rect2, intersection);

    if (intersection.empty()) {
        return 0.0f;
    }

    // 计算交集面积
    float intersection_area = cv::contourArea(intersection);

    // 计算并集面积
    float area1 = rect1.size.width * rect1.size.height;
    float area2 = rect2.size.width * rect2.size.height;
    float union_area = area1 + area2 - intersection_area;

    if (union_area <= 0) {
        return 0.0f;
    }

    return intersection_area / union_area;
}

bool YoloNcnn::isLimitedClass(int classId) const {
    static const int limitedClasses[] = { 1, 2, 3, 4, 7 };
    static const int numLimitedClasses = sizeof(limitedClasses) / sizeof(limitedClasses[0]);

    for (int i = 0; i < numLimitedClasses; ++i) {
        if (limitedClasses[i] == classId) {
            return true;
        }
    }
    return false;
}

// 过滤HeatmapResult结果，特定类别只保留置信度最高的一个
void YoloNcnn::filterMaxOnePerClassHeatmap(std::vector<HeatmapResult>& results) {
    if (results.empty()) {
        return;
    }

    // 使用std::map按类别分组，key是类别ID，value是(索引, 置信度)的向量
    std::map<int, std::vector<std::pair<size_t, float>>> classToResults;

    // 收集所有结果，按类别分组
    for (size_t i = 0; i < results.size(); ++i) {
        int classId = results[i].id;
        float confidence = results[i].confidence;
        classToResults[classId].push_back(std::make_pair(i, confidence));
    }

    // 用于存储需要保留的索引
    std::vector<size_t> keepIndices;
    keepIndices.reserve(results.size());

    // 遍历每个类别
    for (auto& classPair : classToResults) {
        int classId = classPair.first;
        auto& classResults = classPair.second;

        // 如果是需要限制的类别，只保留置信度最高的一个
        if (isLimitedClass(classId)) {
            if (!classResults.empty()) {
                size_t bestIndex = classResults[0].first;
                float bestConfidence = classResults[0].second;

                // 找出置信度最高的索引
                for (size_t i = 1; i < classResults.size(); ++i) {
                    if (classResults[i].second > bestConfidence) {
                        bestIndex = classResults[i].first;
                        bestConfidence = classResults[i].second;
                    }
                }

                keepIndices.push_back(bestIndex);
            }
        }
        else {
            // 对于非限制类别，最多保留置信度最高的两个框
            // 使用 partial_sort 只排序前两个
            int keepCount = std::min(2, static_cast<int>(classResults.size()));
            std::partial_sort(classResults.begin(), classResults.begin() + keepCount, classResults.end(),
                [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                    return a.second > b.second;  // 按置信度降序排列
                });

            // 保留最多前两个
            for (int i = 0; i < keepCount; ++i) {
                keepIndices.push_back(classResults[i].first);
            }
        }
    }

    // 按照keepIndices过滤结果（如果有需要删除的）
    if (keepIndices.size() < results.size()) {
        std::vector<HeatmapResult> filteredResults;
        filteredResults.reserve(keepIndices.size());
        for (size_t idx : keepIndices) {
            filteredResults.push_back(std::move(results[idx]));
        }
        results.swap(filteredResults);
    }

}

void YoloNcnn::postprocessHeatmap(std::vector<HeatmapResult>& output,
    float* data,
    const cv::Vec4d& param) {

    output.clear();

    if (data == nullptr || m_outputMat.empty()) {
        std::cerr << "后处理失败：输出数据为空" << std::endl;
        return;
    }

    const int rawH = m_outputMat.h;
    const int rawW = m_outputMat.w;
    if (rawH <= 0 || rawW <= 0) {
        std::cerr << "后处理失败：输出张量尺寸非法 h=" << rawH
            << ", w=" << rawW << std::endl;
        return;
    }

    cv::Mat rawMat(rawH, rawW, CV_32F, data);
    cv::Mat outputMat = (rawH <= 64 && rawW > rawH) ? rawMat.t() : rawMat;

    const int rows = outputMat.rows;
    const int cols = outputMat.cols;
    std::cout << "[Post] 输出张量: " << rawH << "x" << rawW
        << " -> 解析矩阵: " << rows << "x" << cols
        << ", 流程: YOLOv26_RAW_TOPK"
        << std::endl;

    if (rows <= 0 || cols <= 0) {
        std::cerr << "[Post] 输出行列为空，跳过后处理" << std::endl;
        return;
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::RotatedRect> boxes;
    classIds.reserve(rows);
    confidences.reserve(rows);
    boxes.reserve(rows);

    const int numClasses = cols - 5;
    const int angleIndex = cols - 1;
    if (numClasses <= 0) {
        std::cerr << "[Post] YOLOv26 RAW 输出特征维度不足: " << cols << std::endl;
        return;
    }

    // 预分配临时数组（每行一个槽位，-1 表示无效）
    std::vector<int> tempClassIds(rows, -1);
    std::vector<float> tempConfidences(rows, 0.0f);
    std::vector<cv::RotatedRect> tempBoxes(rows);

    // OpenMP 并行循环
#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        const float* prow = outputMat.ptr<float>(r);

        cv::Mat scores(1, numClasses, CV_32F, const_cast<float*>(prow + 4));
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);

        if (maxClassScore >= m_confidenceThreshold) {
            float cx = prow[0];
            float cy = prow[1];
            float w = prow[2];
            float h = prow[3];
            float angleRad = normalizeAngleToRad(prow[angleIndex]);

            float x = (cx - static_cast<float>(param[2])) / static_cast<float>(param[0]);
            float y = (cy - static_cast<float>(param[3])) / static_cast<float>(param[1]);
            w = w / static_cast<float>(param[0]);
            h = h / static_cast<float>(param[1]);

            if (w > 0.0f && h > 0.0f) {
                tempClassIds[r] = classIdPoint.x;
                tempConfidences[r] = static_cast<float>(maxClassScore);
                tempBoxes[r] = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(w, h),
                    angleRad * 180.0f / static_cast<float>(CV_PI));
            }
        }
    }

    // 串行过滤有效结果
    for (int r = 0; r < rows; ++r) {
        if (tempClassIds[r] >= 0) {
            classIds.push_back(tempClassIds[r]);
            confidences.push_back(tempConfidences[r]);
            boxes.push_back(tempBoxes[r]);
        }
    }

    std::cout << "[Post] 置信度过滤后候选数: " << boxes.size() << std::endl;

    if (!boxes.empty()) {
        std::vector<int> selectedIndices;
        constexpr int kRawTopK = 300;
        selectedIndices.resize(confidences.size());
        std::iota(selectedIndices.begin(), selectedIndices.end(), 0);
        std::sort(selectedIndices.begin(), selectedIndices.end(),
            [&](int a, int b) { return confidences[a] > confidences[b]; });
        if (selectedIndices.size() > kRawTopK) {
            selectedIndices.resize(kRawTopK);
        }

        output.reserve(selectedIndices.size());
        for (int idx : selectedIndices) {
            HeatmapResult result;
            result.id = classIds[idx];
            result.confidence = confidences[idx];

            cv::RotatedRect box = boxes[idx];

            // 暂时不进行坐标转换，等所有处理完成后统一转换
            result.cx = box.center.x;
            result.cy = box.center.y;
            result.l = box.size.width;
            result.s = box.size.height;
            result.angle = box.angle * CV_PI / 180.0f;  // 角度转为弧度

            output.push_back(result);
        }

        std::cout << "[Post] RAW TopK后候选数(k=300): " << output.size() << std::endl;

        // 1：id5与id1、id3、id6有交集则删除
        {
            // 收集所有ID1、ID3和ID6的结果
            std::vector<CachedRotatedRect> id1Rects;
            std::vector<CachedRotatedRect> id3Rects;
            std::vector<CachedRotatedRect> id6Rects;

            for (const auto& result : output) {
                if (result.id == 1) {
                    id1Rects.push_back(makeCachedRect(result));
                }
                else if (result.id == 3) {
                    id3Rects.push_back(makeCachedRect(result));
                }
                else if (result.id == 6) {
                    id6Rects.push_back(makeCachedRect(result));
                }
            }

            // 创建一个新的vector来存储保留的结果
            std::vector<HeatmapResult> filteredResults;

            for (const auto& result : output) {
                bool shouldKeep = true;

                // 只对ID5进行判断
                if (result.id == 5) {
                    const CachedRotatedRect rect5 = makeCachedRect(result);

                    // 检查是否与任何一个ID1有交集
                    for (const auto& rect1 : id1Rects) {
                        if (!boundsOverlap(rect5.bounds, rect1.bounds)) {
                            continue;
                        }

                        float iou_with_id1 = calc_rotate_iou(rect5.rect, rect1.rect);
                        if (iou_with_id1 > 0) {
                            shouldKeep = false;
                            break;
                        }
                    }

                    // 如果还没有被过滤，检查是否与任何一个ID3有交集
                    if (shouldKeep) {
                        for (const auto& rect3 : id3Rects) {
                            if (!boundsOverlap(rect5.bounds, rect3.bounds)) {
                                continue;
                            }

                            float iou_with_id3 = calc_rotate_iou(rect5.rect, rect3.rect);
                            if (iou_with_id3 > 0) {
                                shouldKeep = false;
                                break;
                            }
                        }
                    }

                    // 如果还没有被过滤，检查是否与任何一个ID6有交集
                    if (shouldKeep) {
                        for (const auto& rect6 : id6Rects) {
                            if (!boundsOverlap(rect5.bounds, rect6.bounds)) {
                                continue;
                            }

                            float iou_with_id6 = calc_rotate_iou(rect5.rect, rect6.rect);
                            if (iou_with_id6 > 0) {
                                shouldKeep = false;
                                break;
                            }
                        }
                    }
                }

                if (shouldKeep) {
                    filteredResults.push_back(result);
                }
            }

            // 用过滤后的结果替换原始结果
            output.swap(filteredResults);
        }

        // 3：如果id7和id2都存在，以id7→id2为正方向，若id5在id7的反方向且距离id7大于id7到id2的距离，则删除id5
        {
            // 查找id7和id2
            int id7Index = -1;
            int id2Index = -1;

            for (size_t i = 0; i < output.size(); ++i) {
                if (output[i].id == 7) {
                    id7Index = static_cast<int>(i);
                }
                else if (output[i].id == 2) {
                    id2Index = static_cast<int>(i);
                }
            }

            // 只有当id7和id2都存在时才执行
            if (id7Index >= 0 && id2Index >= 0) {
                const HeatmapResult& id7Result = output[id7Index];
                const HeatmapResult& id2Result = output[id2Index];

                // 计算id7到id2的方向向量）
                float dir_x = id2Result.cx - id7Result.cx;
                float dir_y = id2Result.cy - id7Result.cy;

                // 计算id7到id2的距离
                float dist_7_to_2 = std::sqrt(dir_x * dir_x + dir_y * dir_y);

                // 创建过滤后的结果
                std::vector<HeatmapResult> filteredResults;

                for (const auto& result : output) {
                    bool shouldKeep = true;

                    if (result.id == 5) {
                        // 计算id7指向id5的向量
                        float vec_x = result.cx - id7Result.cx;
                        float vec_y = result.cy - id7Result.cy;

                        // 计算点积来判断方向
                        float dot_product = dir_x * vec_x + dir_y * vec_y;

                        // 如果点积小于0，表示id5在id7的反方向（相对于id7→id2）
                        if (dot_product < 0) {
                            // 计算id5到id7的距离
                            float dist_5_to_7 = std::sqrt(vec_x * vec_x + vec_y * vec_y);

                            // 如果id5到id7的距离大于id7到id2的距离，则删除
                            if (dist_5_to_7 > dist_7_to_2) {
                                shouldKeep = false;
                            }
                        }
                    }

                    if (shouldKeep) {
                        filteredResults.push_back(result);
                    }
                }

                output.swap(filteredResults);
            }
        }



        // 检查是否存在ID7，并找到ID1和ID2
        bool hasId7 = false;
        HeatmapResult id1Result, id2Result;
        bool foundId1 = false, foundId2 = false;

        for (const auto& result : output) {
            if (result.id == 7) {
                hasId7 = true;
            }
            else if (result.id == 1) {
                id1Result = result;
                foundId1 = true;
            }
            else if (result.id == 2) {
                id2Result = result;
                foundId2 = true;
            }
        }

        // 只有在id1和id2都存在的情况下才执行后续
        if (foundId1 && foundId2) {
            // 计算id1指向id2的方向向量
            float dir_x = id2Result.cx - id1Result.cx;
            float dir_y = id2Result.cy - id1Result.cy;

            // 创建一个新的vector来存储保留的结果
            std::vector<HeatmapResult> filteredResults;

            for (const auto& result : output) {
                bool shouldKeep = true;

                // 只对id3和id6进行判断
                if (result.id == 3 || result.id == 6) {
                    // 计算id1指向当前框的向量
                    float vec_x = result.cx - id1Result.cx;
                    float vec_y = result.cy - id1Result.cy;

                    // 计算点积来判断是否在正方向
                    float dot_product = dir_x * vec_x + dir_y * vec_y;

                    // 如果点积大于0，表示在当前框在id1->id2的正方向上
                    if (dot_product > 0) {
                        // 这个框在正方向之后，需要删除
                        shouldKeep = false;
                    }
                }

                // id5如果出现在id1的反方向上，则删除掉
                if (result.id == 5 && shouldKeep) {
                    // 计算id1指向id5的向量
                    float vec_x = result.cx - id1Result.cx;
                    float vec_y = result.cy - id1Result.cy;

                    // 计算点积来判断方向
                    float dot_product = dir_x * vec_x + dir_y * vec_y;

                    // 如果点积小于0，表示在反方向上
                    if (dot_product < 0) {
                        shouldKeep = false;
                    }
                }

                if (shouldKeep) {
                    filteredResults.push_back(result);
                }
            }

            // 用过滤后的结果替换原始结果
            output.swap(filteredResults);

            // 重新查找ID1和ID2
            foundId1 = false;
            foundId2 = false;
            for (const auto& result : output) {
                if (result.id == 1) {
                    id1Result = result;
                    foundId1 = true;
                }
                else if (result.id == 2) {
                    id2Result = result;
                    foundId2 = true;
                }
            }

            // 如果没有检测到ID7，且同时有ID1和ID2，则生成ID7
            if (!hasId7 && foundId1 && foundId2) {
                // 计算ID1和ID2中心点距离
                float dx = id1Result.cx - id2Result.cx;
                float dy = id1Result.cy - id2Result.cy;

                // 生成ID7的位置：在ID1和ID2的正中间
                float new_cx = (id1Result.cx + id2Result.cx) / 2.0f;
                float new_cy = (id1Result.cy + id2Result.cy) / 2.0f;

                // 生成ID7的尺寸：ID1尺寸的80%（面积缩小20%）
                float area_scale = std::sqrt(0.8f); // 面积缩小20%，长宽各乘以sqrt(0.8)
                float new_l = id1Result.l * area_scale;
                float new_s = id1Result.s * area_scale;

                // 创建ID7的结果（使用640x320像素坐标）
                HeatmapResult newId7;
                newId7.id = 7;
                // 使用ID1和ID2置信度的平均值作为ID7的置信度
                newId7.confidence = (id1Result.confidence + id2Result.confidence) / 2.0f;
                newId7.cx = new_cx;  // 640x320像素坐标
                newId7.cy = new_cy;  // 640x320像素坐标
                newId7.l = new_l;    // 640x320像素坐标
                newId7.s = new_s;    // 640x320像素坐标
                newId7.angle = id1Result.angle; // 角度与ID1相同

                output.push_back(newId7);
            }
        }

        // 如果id0与id5有交集，计算有交集的两个id0和id5距离id2和id1的距离，
        // 如果距离id1近，则删除id5，如果距离id2近，则删除id0
        {
            // 收集所有ID0和ID5的结果
            struct CachedResult {
                HeatmapResult result;
                CachedRotatedRect rect;
            };

            std::vector<CachedResult> id0Results;
            std::vector<CachedResult> id5Results;

            for (const auto& result : output) {
                if (result.id == 0) {
                    id0Results.push_back({ result, makeCachedRect(result) });
                }
                else if (result.id == 5) {
                    id5Results.push_back({ result, makeCachedRect(result) });
                }
            }

            // 创建一个新的vector来存储保留的结果
            std::vector<HeatmapResult> filteredResults = output;

            // 检查每个ID0与每个ID5是否有交集
            for (const auto& id0 : id0Results) {
                for (const auto& id5 : id5Results) {
                    if (!boundsOverlap(id0.rect.bounds, id5.rect.bounds)) {
                        continue;
                    }

                    // 检查是否有交集
                    float iou = calc_rotate_iou(id0.rect.rect, id5.rect.rect);
                    if (iou > 0) {
                        // 有交集，需要计算距离
                        // 检查ID1和ID2是否存在
                        if (foundId1 && foundId2) {
                            // 计算id0到id1和id2的距离
                            float dist_id0_to_id1 = std::sqrt(
                                std::pow(id0.result.cx - id1Result.cx, 2) +
                                std::pow(id0.result.cy - id1Result.cy, 2)
                            );
                            float dist_id0_to_id2 = std::sqrt(
                                std::pow(id0.result.cx - id2Result.cx, 2) +
                                std::pow(id0.result.cy - id2Result.cy, 2)
                            );

                            // 计算id5到id1和id2的距离
                            float dist_id5_to_id1 = std::sqrt(
                                std::pow(id5.result.cx - id1Result.cx, 2) +
                                std::pow(id5.result.cy - id1Result.cy, 2)
                            );
                            float dist_id5_to_id2 = std::sqrt(
                                std::pow(id5.result.cx - id2Result.cx, 2) +
                                std::pow(id5.result.cy - id2Result.cy, 2)
                            );

                            // 判断哪个距离更近
                            bool id0_closer_to_id1 = dist_id0_to_id1 < dist_id0_to_id2;
                            bool id5_closer_to_id1 = dist_id5_to_id1 < dist_id5_to_id2;

                            // 情况1：如果id0距离id1更近，删除id5
                            if (id0_closer_to_id1) {
                                // 从filteredResults中删除这个id5
                                auto it = std::remove_if(filteredResults.begin(), filteredResults.end(),
                                    [&](const HeatmapResult& r) {
                                        return r.id == 5 &&
                                            std::abs(r.cx - id5.result.cx) < 0.001f &&
                                            std::abs(r.cy - id5.result.cy) < 0.001f;
                                    }
                                );
                                filteredResults.erase(it, filteredResults.end());
                            }
                            // 情况2：如果id5距离id2更近，删除id0
                            else if (!id5_closer_to_id1) { // id5距离id2更近
                                // 从filteredResults中删除这个id0
                                auto it = std::remove_if(filteredResults.begin(), filteredResults.end(),
                                    [&](const HeatmapResult& r) {
                                        return r.id == 0 &&
                                            std::abs(r.cx - id0.result.cx) < 0.001f &&
                                            std::abs(r.cy - id0.result.cy) < 0.001f;
                                    }
                                );
                                filteredResults.erase(it, filteredResults.end());
                            }
                            // 注意：如果id0距离id2近且id5距离id1近，两者都保留
                        }
                    }
                }
            }

            // 用过滤后的结果替换原始结果
            output.swap(filteredResults);
        }

        // 执行每个类别保留一个结果的过滤
        const size_t beforeClassFilter = output.size();
        filterMaxOnePerClassHeatmap(output);
        std::cout << "[Post] 类别过滤后: "
            << beforeClassFilter << " -> " << output.size() << std::endl;

        if (output.size() < 3) {
            output.clear();
            return;
        }

        // enlargedImage总是是cols*10宽度
        float enlargedWidth = m_heatmapCols * 10.0f;
        float enlargedHeight = 320.0f;

        float scale_x = static_cast<float>(m_heatmapCols) / enlargedWidth;  // 0.1
        float scale_y = 32.0f / enlargedHeight;  // 0.1

        for (auto& result : output) {
            result.cx = result.cx * scale_x;
            result.cy = result.cy * scale_y;
            result.l = result.l * scale_x;
            result.s = result.s * scale_y;
        }
        std::cout << "[Post] 最终输出数量: " << output.size() << std::endl;
        // 删除所有id4的结果（暂时不需要输出）
        output.erase(
            std::remove_if(output.begin(), output.end(),
                [](const HeatmapResult& r) { return r.id == 4; }),
            output.end()
        );

    }
}
