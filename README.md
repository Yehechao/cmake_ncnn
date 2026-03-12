# 安卓so打包
## Build
1.需要安装好ndk才能打包编译
```bash
cd cmake_ncnn
cmake -S . -B build-android-arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=D:/software/AndroidStudioSDK/ndk/29.0.14206865/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DANDROID_STL=c++_static
#若要适配当前工程名，可填加-DANDROID_JNI_CLASS=your/pkg/NcnnApi
cmake --build build-android-arm64 -j8
```
最后输出libncnn_api.so

## 调用借口

## 1. Android Studio 工程内放置文件

以测试App 工程 `D:\yhc_code\MatrixNcnnTest` 为例：

### 1.1 放 `.so`

- `D:\yhc_code\MatrixNcnnTest\app\src\main\jniLibs\arm64-v8a\libncnn_api.so`

说明：

- 当前构建命令是 `c++_static`，通常不需要额外放 `libc++_shared.so`。
- 如果未来改成 `c++_shared`，需要额外放 `libc++_shared.so` 到同目录。

### 1.2 放模型文件

建议放在 assets：

- `D:\yhc_code\MatrixNcnnTest\app\src\main\assets\models\AiBody416n\model.param`
- `D:\yhc_code\MatrixNcnnTest\app\src\main\assets\models\AiBody416n\model.bin`

## 2. Gradle 配置（限制 ABI）

编辑 `app/build.gradle.kts`，在 `defaultConfig` 中加：

```kotlin
ndk {
    abiFilters += "arm64-v8a"
}
```

目的：

- 仅打包 `arm64-v8a`
- 避免其它 ABI 缺少 `.so` 导致安装或运行问题

## 3. JNI 封装类（必须与 JNI 类路径一致）

新建：

- `app/src/main/java/matrix_ncnn/app/NcnnApi.kt`

```kotlin
package matrix_ncnn.app

object NcnnApi {
    init {
        System.loadLibrary("ncnn_api")
    }

    // 兼容旧调用：不传线程数，内部使用默认策略（半核心）
    external fun nativeLoadObbModel(
        paramPath: String,
        binPath: String,
        size: Int,
        conf: Float,
        iou: Float,
        useGpu: Boolean
    ): Boolean

    // 新调用：可显式指定线程数，numThreads <= 0 时仍走默认策略
    external fun nativeLoadObbModel(
        paramPath: String,
        binPath: String,
        size: Int,
        conf: Float,
        iou: Float,
        useGpu: Boolean,
        numThreads: Int
    ): Boolean

    external fun nativeRunObb(
        flatData: FloatArray,
        rows: Int,
        cols: Int
    ): String

    external fun isGpuActive(): Boolean

    external fun nativeRelease()
}
```

JNI 接口签名（当前仓库真实实现）：

- `nativeLoadObbModel(String, String, int, float, float, boolean): boolean`
- `nativeLoadObbModel(String, String, int, float, float, boolean, int): boolean`
- `nativeRunObb(float[], int, int): String`
- `isGpuActive(): boolean`
- `nativeRelease(): void`

## 4. 模型拷贝（assets -> filesDir）

`nativeLoadObbModel` 需要的是“真实文件路径”，不能直接传 `assets/...` 字符串。

可用工具函数：

```kotlin
package matrix_ncnn.app

import android.content.Context
import java.io.File

object AssetUtils {
    fun copyAssetToFiles(context: Context, assetPath: String): String {
        val outFile = File(context.filesDir, assetPath.substringAfterLast('/'))
        if (outFile.exists() && outFile.length() > 0L) return outFile.absolutePath

        context.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return outFile.absolutePath
    }
}
```

## 5. 最小调用流程（Kotlin）

```kotlin
val paramPath = AssetUtils.copyAssetToFiles(this, "models/AiBody416n/model.param")
val binPath = AssetUtils.copyAssetToFiles(this, "models/AiBody416n/model.bin")

val useGpu = true
val loaded = NcnnApi.nativeLoadObbModel(
    paramPath = paramPath,
    binPath = binPath,
    size = 416,
    conf = 0.25f,
    iou = 0.45f,
    useGpu = useGpu
)
check(loaded) { "nativeLoadObbModel failed" }

// 可选：手动指定线程数（例如 6），<=0 时自动按半核心
val loadedWithThreads = NcnnApi.nativeLoadObbModel(
    paramPath = paramPath,
    binPath = binPath,
    size = 416,
    conf = 0.25f,
    iou = 0.45f,
    useGpu = useGpu,
    numThreads = 6
)
check(loadedWithThreads) { "nativeLoadObbModel(with threads) failed" }

// 检测 Vulkan 是否实际启用成功
val gpuActive = NcnnApi.isGpuActive()

val input = FloatArray(32 * 64)
// TODO: 按行优先填充热力图数据: input[r * 64 + c]

val json = NcnnApi.nativeRunObb(input, 32, 64)
// TODO: 用 Gson / kotlinx.serialization 解析 json

// 不再需要时调用
NcnnApi.nativeRelease()
```

## 6. 输出 JSON 结构

成功示例：

```json
{
  "success": true,
  "detections": [
    {
      "id": 5,
      "confidence": 0.968,
      "cx": 15.59,
      "cy": 8.50,
      "l": 21.58,
      "s": 7.25,
      "angle": 0.126
    }
  ]
}
```

失败示例：

```json
{
  "success": false,
  "error": "OBB model not loaded"
}
```

字段说明：

- `angle` 单位是弧度（不是角度）
- `cx/cy/l/s` 与当前 C++ 后处理输出保持一致

## 7. GPU/CPU 策略

- `useGpu = true`：优先 Vulkan，设备不支持会自动回退 CPU
- `useGpu = false`：强制 CPU
- `numThreads <= 0`：自动使用“逻辑核心数的一半”（最少 1）
- `numThreads > 0`：使用调用方指定线程数（最少按 1 兜底）
- `isGpuActive()`：返回当前已加载模型是否真的在用 Vulkan 推理

建议：

- UI 提供一个“GPU 开关”
- 同一设备先跑一次 CPU 和 GPU，记录耗时后再决定默认值

## 8. 常见错误与快速排查

1. `java.lang.UnsatisfiedLinkError: dlopen failed`
- 检查 `libncnn_api.so` 是否在 `app/src/main/jniLibs/arm64-v8a/`
- 检查 `abiFilters` 是否包含 `arm64-v8a`
- 检查安装设备 ABI 是否 `arm64-v8a`

2. `JNI_ERR` 或找不到 native 方法
- 检查 Kotlin 包名是否 `matrix_ncnn.app`
- 检查类名是否 `NcnnApi`
- 检查 JNI 类路径是否仍为 `matrix_ncnn/app/NcnnApi`

3. 模型加载失败
- 检查是否先把 `assets` 复制到了 `filesDir`
- 检查传入的是绝对路径（例如 `.../files/model.param`）

4. 推理返回 `input data length is smaller than rows*cols`
- 检查 `FloatArray` 长度是否 `rows * cols`
- 当前固定输入应为 `32 * 64 = 2048`

5. `useGpu = true` 但 `isGpuActive() = false`
- 设备或驱动不支持 Vulkan，或 Vulkan 初始化未通过
- 当前模型会自动回退 CPU，这属于预期行为

## 9. 注意事项

1. 确认 Android 项目包下存在 `matrix_ncnn.app.NcnnApi`。
2. 确认 `.so` 放在 `app/src/main/jniLibs/arm64-v8a/`。
3. 确认模型在 `assets/models/AiBody416n/`。
4. 确认先复制模型到 `filesDir` 再调用 `nativeLoadObbModel`。
5. 确认输入尺寸固定 `rows=32, cols=64`。
6. 如需控制线程，改用 `nativeLoadObbModel(..., numThreads)`；不传则默认半核心。
7. 需要确认 GPU 是否真的启用时，调用 `isGpuActive()`。
8. 若修改包名，必须同步修改 CMake `ANDROID_JNI_CLASS` 并重新编译 `.so`。
