# SO 接口交接说明（当前版本）

## 1) 目的
该文档仅描述当前 `.so` 可调用接口与数据协议，供另一个 Codex 直接接入。  
不包含内部实现细节。

## 2) JNI 对外接口（Kotlin/Java）
- `nativeLoadObbModel(paramPath: String, binPath: String, size: Int, conf: Float, iou: Float, useGpu: Boolean): Boolean`
- `nativeLoadObbModel(paramPath: String, binPath: String, size: Int, conf: Float, iou: Float, useGpu: Boolean, numThreads: Int): Boolean`
- `nativeRunObb(flatData: FloatArray, rows: Int, cols: Int): FloatArray`
- `isGpuActive(): Boolean`
- `nativeRelease(): Unit`

## 3) `nativeRunObb` 协议
- 输入：
  - `flatData`：`float32` 扁平数组，行优先排列
  - `rows/cols`：输入尺寸（当前常用 `32 x 64`）
  - 约束：`flatData.size >= rows * cols`
- 输出：结构化 `FloatArray`
  - `packed[0]`：`success`（`1.0` 成功，`0.0` 失败）
  - `packed[1]`：`count`（检测数量）
  - 从 `packed[2]` 开始，每个检测 7 个字段：
    - `id, confidence, cx, cy, l, s, angle`

## 4) 调用方解析规则
- 第 `i` 个检测偏移：`base = 2 + i * 7`
- 字段读取：
  - `id = packed[base + 0].toInt()`
  - `confidence = packed[base + 1]`
  - `cx = packed[base + 2]`
  - `cy = packed[base + 3]`
  - `l = packed[base + 4]`
  - `s = packed[base + 5]`
  - `angle = packed[base + 6]`

## 5) 当前交付边界
- 当前正式接口为 `FloatArray -> FloatArray` 结构化返回版本。
- 仅保留 OBB/Cls 推理主链路。
- 与推理无关的画图、轮廓提取、可视化代码已移除。

## 6) Android 端查看 `.so` 日志（模型加载输出）
- 结论：
  - `YoloInference.cpp` 里的 `std::cout/std::cerr` 在 Android App 进程里**不保证稳定可见**。
  - 如果要稳定在 Logcat 看到模型加载输出，建议使用 `__android_log_print`。

- 推荐做法（native）：
  - 在 C++ 中引入：
    - `#include <android/log.h>`
  - 使用：
    - `__android_log_print(ANDROID_LOG_INFO, "ncnn_api", "输入尺寸: %dx%d", m_netWidth, m_netHeight);`
  - 错误日志用：
    - `__android_log_print(ANDROID_LOG_ERROR, "ncnn_api", "...");`

- Android Studio 查看：
  - 打开 Logcat，过滤 `tag:ncnn_api`。

- 命令行查看：
  - `adb logcat -s ncnn_api`

- 建议：
  - 把模型加载成功、线程数、GPU 开关、`isGpuActive` 结果都打到同一 tag，便于联调定位。
