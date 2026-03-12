#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// 热力图推理结果
struct HeatmapResult {
    int id;           // 类别ID
    float confidence; // 置信度
    float cx;         // 中心x坐标（相对于输入热力图的列数）
    float cy;         // 中心y坐标（相对于输入热力图的行数，固定32行）
    float l;          // 长边长度（相对于输入热力图尺寸）
    float s;          // 短边长度（相对于输入热力图尺寸）
    float angle;      // 旋转角度（弧度）
};

// 分类推理结果
struct ClassifyResult {
    int classId;        // 分类ID（姿势ID）
    float confidence;   // 置信度

    ClassifyResult() : classId(-1), confidence(0.0f) {}
    ClassifyResult(int id, float conf) : classId(id), confidence(conf) {}
};

