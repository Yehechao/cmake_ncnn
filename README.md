## windows第三方版本下载地址

通过网盘分享的文件：ncnn_3rdpart.rar
链接: https://pan.baidu.com/s/1j-y2Ux4wGeCS15Fq_JxG1w?pwd=9h4b 提取码: 9h4b

## CMake 编译


配置：

## windows第三方版本下载地址

通过网盘分享的文件：ncnn_3rdpart.rar
链接: https://pan.baidu.com/s/1j-y2Ux4wGeCS15Fq_JxG1w?pwd=9h4b 提取码: 9h4b

## CMake 编译 EXE

配置：

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64
```

Release 编译：

```powershell
cmake --build build-cmake --config Release
```

编译完成后，可执行文件默认在：

```text
cmake_ncnn.exe
```

## CMake 编译 DLL

默认只编译 EXE。要编译 DLL，请在配置时打开 `BUILD_CMAKE_NCNN_DLL`。

仅编译 DLL（不编译 EXE）：

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CMAKE_NCNN_EXE=OFF -DBUILD_CMAKE_NCNN_DLL=ON
cmake --build build-cmake --config Release
```

同时编译 EXE + DLL：

```powershell
cmake -S . -B build-cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CMAKE_NCNN_EXE=ON -DBUILD_CMAKE_NCNN_DLL=ON
cmake --build build-cmake --config Release
```

编译完成后，默认输出到项目根目录：

```text
cmake_ncnn.dll
cmake_ncnn.lib
```
