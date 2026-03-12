## windows第三方版本下载地址

通过网盘分享的文件：ncnn_3rdpart.rar
链接: https://pan.baidu.com/s/1j-y2Ux4wGeCS15Fq_JxG1w?pwd=9h4b 提取码: 9h4b

## 编译 exe：
```bash
cmake -S . -B build-exe -DBUILD_NCNN_DLL=OFF
cmake --build build-exe --config Release
```
## 编译 dll：
```bash
cmake -S . -B build-dll -DBUILD_NCNN_DLL=ON
cmake --build build-dll --config Release
```
