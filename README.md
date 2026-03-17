# 安卓 `.so` 打包与调用说明（当前实现）

本文档对应当前仓库代码，目标是让你可以：
1. 编译 `libncnn_api.so`
2. 在 Android 项目中接入并调用
3. 正确查看 native（`.so`）日志

## 1. 编译 `libncnn_api.so`

在仓库根目录执行（示例 `arm64-v8a`）：

```bash
cmake -S . -B build-android-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DANDROID_STL=c++_static
cmake --build build-android-arm64 -j8
```

输出文件：
- `build-android-arm64/libncnn_api.so`

如果 Java/Kotlin 类路径不是 `matrix_ncnn.app.NcnnApi`，编译时追加：

```bash
-DANDROID_JNI_CLASS=your/pkg/NcnnApi
```

## 2. 放入 Android 工程

- `.so` 放到：`app/src/main/jniLibs/arm64-v8a/libncnn_api.so`
- 模型放到：`app/src/main/assets/models/<model_dir>/model.param` 和 `model.bin`
- 调用前需先把模型从 assets 拷贝到 `filesDir`，然后把绝对路径传给 native

## 3. JNI 调用接口（当前）

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

    // 返回: [success, count, det0(7), det1(7), ...]
    external fun nativeRunObb(flatData: FloatArray, rows: Int, cols: Int): FloatArray

    external fun isGpuActive(): Boolean
    external fun nativeRelease()
}
```

## 4. `nativeRunObb` 返回协议

- `packed[0]`：`success`（`1.0` 成功，`0.0` 失败）
- `packed[1]`：`count`
- 每个检测固定 7 个字段：`id, confidence, cx, cy, l, s, angle`
- 第 `i` 个检测起始下标：`base = 2 + i * 7`

说明：
- 当前 JNI 输出缓冲按 `maxDetections=64` 保护边界，超出部分会截断（这是缓冲上限，不是类别限制）。

## 5. 最小调用示例

```kotlin
val ok = NcnnApi.nativeLoadObbModel(
    paramPath = paramPath,
    binPath = binPath,
    size = 416,
    conf = 0.25f,
    iou = 0.45f,
    useGpu = false,
    numThreads = 2
)
check(ok)

val packed = NcnnApi.nativeRunObb(input, rows = 32, cols = 64)
val success = packed.isNotEmpty() && packed[0] > 0.5f
val count = if (packed.size >= 2) packed[1].toInt() else 0
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
    }
}
NcnnApi.nativeRelease()
```

## 6. 日志与排查（重点）

当前 native 模型加载与推理日志来自 `.so`（`YoloInference.cpp`），走 Android `logcat`：
- tag: `ncnn_api`

Android 调用层日志：
- `NcnnApi.kt` 加载 so：tag `NcnnApiLoader`
- `MainActivity.kt` 模型加载流程：tag `MatrixNcnnLoad`

建议直接看：

```bash
adb logcat -s ncnn_api NcnnApiLoader MatrixNcnnLoad
```

如果“App 页面日志”看不到 `.so` 加载日志，优先检查：
1. `jniLibs` 是否是最新 `libncnn_api.so`（避免使用旧 so）。
2. 你的页面日志过滤规则是否包含 `ncnn_api` 关键字。

## 7. 常见问题

1. `UnsatisfiedLinkError`
- 检查 so 放置路径和 ABI（`arm64-v8a`）是否一致。

2. `JNI_ERR` / 找不到 native 方法
- 确认包名类名与 `-DANDROID_JNI_CLASS` 一致。

3. `useGpu=true` 但 `isGpuActive=false`
- 设备不支持或 Vulkan 初始化失败时会自动回退 CPU，属于预期行为。
