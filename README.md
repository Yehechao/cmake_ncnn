# 安卓 `.so` 编译与接口调用指南

本文档用于：  
1) 从源码编译 `libncnn_api.so`  
2) 在 Android 工程中正确调用当前接口（`FloatArray -> FloatArray` 结构化返回）

## 1. 环境准备

- 系统：Windows
- CMake：>= 3.15
- Ninja：已安装并可执行
- Android NDK：建议 29.x（需包含 `android.toolchain.cmake`）

请确认仓库内第三方库目录存在：
- `ncnn_3rdpart/ncnn-20260113-android-vulkan/<ABI>/...`
- `ncnn_3rdpart/opencv-mobile-4.8.0-android/sdk/native/...`

## 2. 编译 `libncnn_api.so`

在仓库根目录执行（示例 `arm64-v8a`）：

```bash
cmake -S . -B build-android-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=D:/software/AndroidStudioSDK/ndk/29.0.14206865/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DANDROID_STL=c++_static
cmake --build build-android-arm64 -j8
```

输出：
- `build-android-arm64/libncnn_api.so`（或等价目标目录）

如果你的 Android 包名/类名不是 `matrix_ncnn.app.NcnnApi`，编译时加：

```bash
-DANDROID_JNI_CLASS=your/pkg/NcnnApi
```

## 3. Android 工程放置文件

### 3.1 放 `.so`
- `app/src/main/jniLibs/arm64-v8a/libncnn_api.so`

### 3.2 放模型
- `app/src/main/assets/models/AiBody416n/model.param`
- `app/src/main/assets/models/AiBody416n/model.bin`

## 4. JNI 接口签名（当前实现）

```kotlin
object NcnnApi {
    init { System.loadLibrary("ncnn_api") }

    external fun nativeLoadObbModel(
        paramPath: String,
        binPath: String,
        size: Int,
        conf: Float,
        iou: Float,
        useGpu: Boolean
    ): Boolean

    external fun nativeLoadObbModel(
        paramPath: String,
        binPath: String,
        size: Int,
        conf: Float,
        iou: Float,
        useGpu: Boolean,
        numThreads: Int
    ): Boolean

    // 返回结构化 FloatArray：[success, count, det0(7), det1(7), ...]
    external fun nativeRunObb(
        flatData: FloatArray,
        rows: Int,
        cols: Int
    ): FloatArray

    external fun isGpuActive(): Boolean
    external fun nativeRelease()
}
```

## 5. `nativeRunObb` 输入/输出协议

### 输入
- `flatData`：按行优先 `float32`（`flatData[r * cols + c]`）
- 长度至少 `rows * cols`（当前常用 `32 * 64 = 2048`）

### 输出（结构化 `FloatArray`）
- `packed[0]`：`success`（`1.0` 成功，`0.0` 失败）
- `packed[1]`：`count`（检测数量）
- 从 `packed[2]` 开始，每个检测固定 7 个字段：
  - `id, confidence, cx, cy, l, s, angle`

第 `i` 个检测起始偏移：
- `base = 2 + i * 7`

这里“偏移”是指：第 `i` 个检测在 `packed` 数组中的起始下标。  
示例：
- 第 0 个检测：`base = 2`
- 第 1 个检测：`base = 9`
- 第 2 个检测：`base = 16`

然后按顺序读取：
- `packed[base + 0]` -> `id`
- `packed[base + 1]` -> `confidence`
- `packed[base + 2]` -> `cx`
- `packed[base + 3]` -> `cy`
- `packed[base + 4]` -> `l`
- `packed[base + 5]` -> `s`
- `packed[base + 6]` -> `angle`

## 6. 最小调用示例（Kotlin）

```kotlin
val paramPath = /* assets 拷贝到 filesDir 后的绝对路径 */
val binPath = /* assets 拷贝到 filesDir 后的绝对路径 */

val loaded = NcnnApi.nativeLoadObbModel(
    paramPath = paramPath,
    binPath = binPath,
    size = 416,
    conf = 0.25f,
    iou = 0.45f,
    useGpu = true,
    numThreads = 4
)
check(loaded) { "nativeLoadObbModel failed" }

val rows = 32
val cols = 64
val input = FloatArray(rows * cols)
// TODO: 按行优先填充 input[r * cols + c]

val packed = NcnnApi.nativeRunObb(input, rows, cols)
val success = packed.isNotEmpty() && packed[0] > 0.5f
val count = if (packed.size >= 2) packed[1].toInt() else 0

if (success) {
    for (i in 0 until count) {
        val base = 2 + i * 7
        if (base + 7 <= packed.size) {
            val id = packed[base + 0].toInt()
            val conf = packed[base + 1]
            val cx = packed[base + 2]
            val cy = packed[base + 3]
            val l = packed[base + 4]
            val s = packed[base + 5]
            val angle = packed[base + 6]
            // TODO: 使用检测结果
        }
    }
}

NcnnApi.nativeRelease()
```

## 7. 常见问题

1. `UnsatisfiedLinkError`
- 检查 `.so` 是否在 `jniLibs/arm64-v8a/`
- 检查 `abiFilters` 是否包含 `arm64-v8a`

2. `JNI_ERR` 或找不到 native 方法
- 检查 Kotlin 包名/类名与编译时 JNI 类路径一致
- 若包名变更，重新编译并传 `-DANDROID_JNI_CLASS=...`

3. `nativeRunObb` 返回失败
- 检查 `flatData.size >= rows * cols`
- 检查是否先成功调用 `nativeLoadObbModel`

4. `useGpu=true` 但 `isGpuActive()==false`
- 设备不支持或初始化失败，自动回退 CPU，属预期行为

## 8. 当前代码主链路说明

- 工程已精简为 OBB/Cls 推理主链路。
- 与推理无关的画图、轮廓提取、可视化叠加代码已移除，不再影响推理路径。
