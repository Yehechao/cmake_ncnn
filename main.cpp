#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "ObjectDetectInference.h"
using namespace std;
using namespace chrono;
namespace fs = filesystem;

vector<vector<float>> readTxtFile(const string& filePath) {
    vector<vector<float>> data(32, vector<float>(64, 0.0f));
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "错误: 无法打开文件 - " << filePath << endl;
        return data;
    }
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();

    content.erase(remove(content.begin(), content.end(), ' '), content.end());
    content.erase(remove(content.begin(), content.end(), '\t'), content.end());
    content.erase(remove(content.begin(), content.end(), '\r'), content.end());
    content.erase(remove(content.begin(), content.end(), '\n'), content.end());

    if (content.length() < 4 || content.substr(0, 2) != "{{" || content.substr(content.length() - 2) != "}}") {
        cerr << "错误: 文件格式不正确" << endl;
        return data;
    }

    content = content.substr(2, content.length() - 4);
    vector<string> rows;
    size_t start = 0, end;
    while ((end = content.find("},{", start)) != string::npos) {
        rows.push_back(content.substr(start, end - start));
        start = end + 3;
    }
    rows.push_back(content.substr(start));

    if (rows.size() != 32) {
        cerr << "错误: 文件应该有32行数据，实际找到" << rows.size() << "行" << endl;
        return data;
    }

    for (size_t rowIdx = 0; rowIdx < rows.size(); ++rowIdx) {
        string rowStr = rows[rowIdx];
        if (rowStr.front() == '{') rowStr = rowStr.substr(1);
        if (rowStr.back() == '}') rowStr = rowStr.substr(0, rowStr.length() - 1);
        vector<float> rowData;
        stringstream ss(rowStr);
        string value;
        while (getline(ss, value, ',')) {
            try {
                rowData.push_back(stof(value));
            }
            catch (...) {
                rowData.push_back(0.0f);
            }
        }

        if (rowData.size() != 64) {
            cerr << "警告: 行 " << rowIdx + 1 << " 应该有64个数据，实际找到" << rowData.size() << "个" << endl;
        }

        for (size_t colIdx = 0; colIdx < min(rowData.size(), (size_t)64); ++colIdx) {
            data[rowIdx][colIdx] = rowData[colIdx];
        }
    }
    return data;
}

vector<string> getTxtFileList(const string& folderPath) {
    vector<string> txtFiles;
    try {
        if (!fs::exists(folderPath)) {
            cerr << "错误: 文件夹不存在 - " << folderPath << endl;
            return txtFiles;
        }
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                txtFiles.push_back(entry.path().string());
            }
        }
        sort(txtFiles.begin(), txtFiles.end(), [](const string& a, const string& b) {
            string nameA = fs::path(a).stem().string();
            string nameB = fs::path(b).stem().string();
            try { if (stoi(nameA) != stoi(nameB)) return stoi(nameA) < stoi(nameB); }
            catch (...) {}
            if (nameA.length() != nameB.length()) return nameA.length() < nameB.length();
            return nameA < nameB;
            });
        cout << "找到 " << txtFiles.size() << " 个txt文件" << endl;
    }
    catch (const fs::filesystem_error& e) {
        cerr << "文件系统错误: " << e.what() << endl;
    }
    return txtFiles;
}

const char* getPoseName(int classId) {
    static const char* poseNames[] = {
        "Lying_face_down", "lie_down"
    };
    if (classId >= 0 && classId < 2) {
        return poseNames[classId];
    }
    return "未知姿势";
}

void createOutputFolder(const string& folderPath) {
    try {
        if (!fs::exists(folderPath)) {
            fs::create_directories(folderPath);
            cout << "创建结果文件夹: " << folderPath << endl;
        }
    }
    catch (const fs::filesystem_error& e) {
        cerr << "无法创建结果文件夹: " << e.what() << endl;
    }
}


int main() {

    // NCNN 模型路径
    string clsParamPath = "./models/AiPostrue224n/model.param";
    string clsBinPath = "./models/AiPostrue224n/model.bin";
    string obbParamPath = "./models/AiBody416n/model.param";
    string obbBinPath = "./models/AiBody416n/model.bin";
    string input_folder = "./data";
    string output_folder = "./results";
    int imgsz_obb = 416;
    int imgsz_cls = 224;
    int mode = 2;

    vector<string> txtFiles = getTxtFileList(input_folder);
    long long totalForwardTime = 0;
    int successfulInferences = 0;
    //单独肢体检测
    if (mode == 1) {
        // 加载OBB旋转框检测模型
        auto model = YoloNcnn::load_obb(obbParamPath, obbBinPath, imgsz_obb, 0.25, 0.45,4);

        for (size_t fileIndex = 0; fileIndex < txtFiles.size(); ++fileIndex) {
            string filePath = txtFiles[fileIndex];
            string fileName = fs::path(filePath).filename().string();
            string baseName = fs::path(filePath).stem().string();

            // 读取热力图txt数据（32x64）
            vector<vector<float>> heatmapData2D = readTxtFile(filePath);
            vector<HeatmapResult> obbResults;   // 后处理后的检测结果

            // 推理计时
            auto start_forward = high_resolution_clock::now();
            bool success = model->run(obbResults, heatmapData2D);
            auto end_forward = high_resolution_clock::now();
            auto forwardDuration = duration_cast<microseconds>(end_forward - start_forward);
            cout << " [" << fileIndex + 1 << "/" << txtFiles.size() << "] 推理耗时: " << forwardDuration.count() / 1000.0 << " ms" << endl;

            // 生成热力图可视化并绘制检测结果
            cv::Mat heatmapImg = model->createHeatmapImageFromData(heatmapData2D, true, 0.03);

            // 此行代码注释，解开下方 drawPredOnHeatmap(heatmapImg, obbResults) 可以去除轮廓提取，画方框蒙版
            auto contours = model->extractContours(heatmapData2D, 10);
            // 提取热力图轮廓 model->drawPredOnHeatmap(heatmapImg, obbResults, contours);
            // model->drawPredOnHeatmap(heatmapImg, obbResults)
            // 保存结果图片
            createOutputFolder(output_folder);
            fs::path outPath = fs::path(output_folder) / (baseName + ".jpg");
            cv::imwrite(outPath.string(), heatmapImg);

            if (success) {
                totalForwardTime += forwardDuration.count();
                successfulInferences++;
            }
        }

        if (successfulInferences > 0) {
            cout << "平均耗时: " << fixed << setprecision(2)
                << (totalForwardTime / 1000.0 / successfulInferences) << " ms" << endl;
        }

    }
    //肢体检测+睡姿检测
    else if (mode == 2) {
        // 加载分类模型（睡姿识别）和OBB检测模型
        auto clsModel = YoloNcnn::load_cls(clsParamPath, clsBinPath, imgsz_cls, 4);
        auto model = YoloNcnn::load_obb(obbParamPath, obbBinPath, imgsz_obb, 0.25, 0.45, 4);

        for (size_t fileIndex = 0; fileIndex < txtFiles.size(); ++fileIndex) {
            string filePath = txtFiles[fileIndex];
            string fileName = fs::path(filePath).filename().string();
            string baseName = fs::path(filePath).stem().string();

            // 读取热力图txt数据（32x64）
            vector<vector<float>> heatmapData2D = readTxtFile(filePath);
            vector<HeatmapResult> obbResults;   // 后处理后的检测结果
            ClassifyResult poseResult;           // 睡姿分类结果

            // 推理计时：OBB检测 + 睡姿分类 + 轮廓提取
            auto start_forward = high_resolution_clock::now();
            bool success = model->forward(clsModel, heatmapData2D, obbResults, poseResult, true, 0.03f);
            auto contours = model->extractContours(heatmapData2D, 10);  // 提取热力图轮廓
            auto end_forward = high_resolution_clock::now();
            auto forwardDuration = duration_cast<microseconds>(end_forward - start_forward);
            cout << " [" << fileIndex + 1 << "/" << txtFiles.size() << "] 推理耗时: " << forwardDuration.count() / 1000.0 << " ms" << endl;



            // 绘制并保存结果图像
            cv::Mat heatmapImg = model->createHeatmapImageFromData(heatmapData2D, true, 0.03);
            if (!heatmapImg.empty()) {

                model->drawPredOnHeatmap(heatmapImg, obbResults, heatmapData2D, contours);
                // 在图像上标注睡姿
                string poseLabel = "Pose: " + string(getPoseName(poseResult.classId)) +
                    " (" + to_string(poseResult.classId) + ")";
                cv::putText(heatmapImg, poseLabel, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
                createOutputFolder(output_folder);
                fs::path outPath = fs::path(output_folder) / (baseName + ".png");
                cv::imwrite(outPath.string(), heatmapImg);
            }

            // 统计推理耗时
            if (success) {
                totalForwardTime += forwardDuration.count();
                successfulInferences++;
            }
        }

        // 输出平均耗时统计
        if (successfulInferences > 0) {
            cout << "平均耗时: " << fixed << setprecision(2)
                << (totalForwardTime / 1000.0 / successfulInferences) << " ms" << endl;
        }
    }

    return 0;
}
