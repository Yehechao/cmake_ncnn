## windows第三方版本下载地址

通过网盘分享的文件：ncnn_3rdpart.rar
链接: https://pan.baidu.com/s/1j-y2Ux4wGeCS15Fq_JxG1w?pwd=9h4b 提取码: 9h4b

## CMake 编译

### 仅编译 EXE

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CMAKE_NCNN_EXE=ON -DBUILD_CMAKE_NCNN_DLL=OFF
cmake --build build-cmake --config Release
```

输出：`cmake_ncnn.exe`

### 仅编译 DLL

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CMAKE_NCNN_EXE=OFF -DBUILD_CMAKE_NCNN_DLL=ON
cmake --build build-cmake --config Release
```

输出：`cmake_ncnn.dll` + `cmake_ncnn.lib`

### 同时编译 EXE + DLL

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CMAKE_NCNN_EXE=ON -DBUILD_CMAKE_NCNN_DLL=ON
cmake --build build-cmake --config Release
```

输出：`cmake_ncnn.exe` + `cmake_ncnn.dll` + `cmake_ncnn.lib`

### DLL 使用说明

只需要以下文件：
- 头文件：`ObjectDetectInference.h`、`utils.h`
- 导入库：`cmake_ncnn.lib`
- 动态库：`cmake_ncnn.dll`

## Release 优化选项

与 Visual Studio 项目保持一致：

- 编译：`/O2 /Oi /Gy /GL`
- 链接：`/LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO`
