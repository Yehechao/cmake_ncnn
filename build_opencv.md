# OpenCV WASM 最小编译说明

目标：

- 只编译当前工程实际需要的 OpenCV 模块
- 不启用 SIMD
- 不启用 pthread
- 所有命令都保持单行，方便直接粘贴到 PowerShell

当前工程实际使用的 OpenCV 静态库只有：

- `opencv_core`
- `opencv_imgproc`
- `opencv_imgcodecs`

同时，`wasm/CMakeLists.txt` 还会直接链接：

- `ncnn_3rdpart/opencv_wasm_build/3rdparty/lib/libzlib.a`
- `ncnn_3rdpart/opencv_wasm_build/3rdparty/lib/liblibpng.a`
- `ncnn_3rdpart/opencv_wasm_build/3rdparty/lib/liblibjpeg-turbo.a`

所以这次最小编译仍然需要同时保留：

- `ncnn_3rdpart/opencv_wasm_build`
- `ncnn_3rdpart/opencv_wasm_install`

这次不要再走 `platforms/js/build_js.py`，因为它在当前机器上容易复用到 `Visual Studio 17 2022` 生成器，最后把 Emscripten 参数喂给 `cl.exe`，从而触发你看到的 baseline 报错。

下面这套 `emcmake cmake -G Ninja` 命令，我已经在当前机器上验证过，`configure` 可以通过。

## 1. 先加载 Emscripten 环境

```powershell
cd D:\yhc_code\emsdk-5.0.0; .\emsdk_env.ps1; cd D:\yhc_code\ncnn_cpu\cmake_ncnn
```

## 2. 删除旧目录

```powershell
Remove-Item -Recurse -Force .\ncnn_3rdpart\opencv_wasm_build,.\ncnn_3rdpart\opencv_wasm_install -ErrorAction SilentlyContinue
```

## 3. 配置最小 OpenCV WASM

```powershell
cmd /c "emcmake cmake -G Ninja -S .\ncnn_3rdpart\opencv\sources -B .\ncnn_3rdpart\opencv_wasm_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=D:/yhc_code/ncnn_cpu/cmake_ncnn/ncnn_3rdpart/opencv_wasm_install -DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_js=OFF -DBUILD_opencv_world=OFF -DBUILD_JAVA=OFF -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DWITH_JPEG=ON -DWITH_WEBP=OFF -DWITH_TIFF=OFF -DWITH_OPENJPEG=OFF -DWITH_OPENEXR=OFF -DWITH_QUIRC=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF -DWITH_ADE=OFF -DWITH_OPENMP=OFF -DWITH_PTHREADS_PF=OFF -DCV_ENABLE_INTRINSICS=OFF -DCV_DISABLE_OPTIMIZATION=ON -DCPU_BASELINE= -DCPU_DISPATCH="
```

说明：

- `-G Ninja` 是关键，避免再落回 Visual Studio 生成器
- `CV_ENABLE_INTRINSICS=OFF` 明确关闭 intrinsics
- `CV_DISABLE_OPTIMIZATION=ON` 直接关掉 OpenCV 这套优化探测，避免 baseline 空值检查再报错
- `BUILD_LIST=core,imgproc,imgcodecs` 就是当前工程需要的最小模块集

## 4. 编译

```powershell
cmd /c "emmake ninja -C .\ncnn_3rdpart\opencv_wasm_build -j8"
```

## 5. 安装

```powershell
cmd /c "cmake --install .\ncnn_3rdpart\opencv_wasm_build"
```

## 6. 编译完成后应看到的关键文件

安装目录：

```text
ncnn_3rdpart/opencv_wasm_install/
├─ include/opencv4/opencv2/core
├─ include/opencv4/opencv2/imgproc
├─ include/opencv4/opencv2/imgcodecs
└─ lib
   ├─ libopencv_core.a
   ├─ libopencv_imgproc.a
   ├─ libopencv_imgcodecs.a
   └─ cmake/opencv4/...
```

构建目录：

```text
ncnn_3rdpart/opencv_wasm_build/
└─ 3rdparty/lib
   ├─ libzlib.a
   ├─ liblibpng.a
   └─ liblibjpeg-turbo.a
```

## 7. 为什么这版是“最小编译”

原因只有两点：

1. 只构建了 `core,imgproc,imgcodecs`
2. 关闭了 tests/examples/docs/apps/js/world，以及不需要的图片和视频后端

也就是说，这版不是完整 OpenCV wasm，而是专门给当前工程使用的最小静态库集合。

## 8. 当前建议

如果你下一步只是想先把 wasm 工程编过：

1. 先按这份文档重新生成 `opencv_wasm_build` 和 `opencv_wasm_install`
2. 再去构建 `wasm/`
3. 暂时不要再开 SIMD，先确认 basic 版本链路稳定
