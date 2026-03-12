# 编译和运行 WebAssembly 项目

## 1. 进入项目目录并创建构建目录
```bash
cd 当前目录
cd wasm          # wasm 的 CMakeLists.txt 位于 wasm 目录下
mkdir build
cd build
```
## 2. 使用 Emscripten 配置 CMake
```
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -G "Ninja" 
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -G "Ninja" -DWASM_USE_SIMD=OFF -DWASM_USE_PTHREADS=OFF # 显式关闭 SIMD 和线程
```
## 3. 编译项目
```
emmake ninja -j $ (nproc)   # 将 `-jx` 中的 `x` 替换为你的 CPU 核心数，或使用 `$(nproc)` 自动获取

```

# WASM API 接口文档

本文档对应当前精简后的 WASM 导出：仅保留 OBB 模型加载与推理。

## 调用示例

```js

// 1) 加载 OBB 模型
const ok = Module.loadObbModel('/model_obb.param', '/model_obb.bin', 416, 0.25, 0.45);
if (!ok) throw new Error('loadObbModel failed');

// 2) 推理（输入为 32x64 的一维 Float32Array）
const jsonStr = Module.runObb(flatData, 32, 64);
const result = JSON.parse(jsonStr);
console.log(result.detections);
```

## 导出接口

### 1. `loadObbModel(paramPath, binPath, size, conf, iou) -> bool`
- 功能：加载 OBB NCNN 模型。
- 参数：
  - `paramPath`: `.param` 文件路径
  - `binPath`: `.bin` 文件路径
  - `size`: 模型输入尺寸（例如 `416`）
  - `conf`: 置信度阈值（例如 `0.25`）
  - `iou`: NMS 阈值（例如 `0.45`）

### 2. `runObb(flatData, rows, cols) -> string(JSON)`
- 功能：执行 OBB 推理。
- 参数：
  - `flatData`: 一维 `Float32Array`
  - `rows`: 热力图行数（通常 `32`）
  - `cols`: 热力图列数（通常 `64`）
- 返回：JSON 字符串。

## 返回 JSON

成功示例：

```jsonc
{
  "success": true, // 是否推理成功
  "detections": [
    {
      "id": 5, // 类别ID（人体部位ID）
      "confidence": 0.968, // 置信度
      "cx": 15.59, // 中心点X（相对于输入热力图列数）
      "cy": 8.50, // 中心点Y（相对于输入热力图行数）
      "l": 21.58, // 长边长度（相对于输入热力图尺寸）
      "s": 7.25, // 短边长度（相对于输入热力图尺寸）
      "angle": 0.126 // 旋转角（弧度）
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

`detections` 字段说明：
- `id`：类别ID（人体部位ID）
- `confidence`：置信度
- `cx`：中心点X（相对于输入热力图列数）
- `cy`：中心点Y（相对于输入热力图行数）
- `l`：长边长度（相对于输入热力图尺寸）
- `s`：短边长度（相对于输入热力图尺寸）
- `angle`：旋转角，单位是**弧度**（不是角度）

若前端需要“角度(度)”显示，可用：
- `angleDeg = angle * 180 / Math.PI`
