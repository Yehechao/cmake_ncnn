# HANDOFF SUMMARY

## 1. 交接目标
- 安卓项目 `D:\yhc_code\ncnn_cpu\cmake_ncnn\` 已迁移为**纯 YOLOv26 OBB 后处理**。
- 按你的最新要求完成：
  - 去掉旧分支兼容。
  - 去掉同类/跨类 NMS。
  - 保留业务规则过滤链（各 ID 关系规则）。
  - 保留“不同类别保留不同数量”的逻辑。

## 2. 本次实际改动文件
- `D:\yhc_code\ncnn_cpu\cmake_ncnn\ObjectDetectInference.h`
- `D:\yhc_code\ncnn_cpu\cmake_ncnn\Post.cpp`

## 3. 关键改动说明

### 3.1 头文件接口清理（去 NMS 相关声明）
- 已删除以下声明：
  - `convariance_matrix(...)`
  - `box_probiou(...)`
  - `rotate_nms(...)`
- 保留：
  - `calc_rotate_iou(...)`
  - `filterMaxOnePerClassHeatmap(...)`
  - `isLimitedClass(...)`

### 3.2 后处理固定为单一路径（无分支）
- `postprocessHeatmap(...)` 改为固定流程：
  - 日志固定显示：`流程: YOLOv26_RAW_TOPK`
  - 不再存在 LEGACY/旧模型分支。

### 3.3 解析与筛选逻辑（纯 YOLOv26 RAW）
- 按 RAW 输出解析：
  - `numClasses = cols - 5`
  - `angleIndex = cols - 1`
- 置信度过滤后执行 TopK：
  - 固定 `k = 300`
  - 不做同类 NMS、不做跨类 NMS

### 3.4 业务过滤链与类别保留逻辑
- 完整保留原有业务规则链（ID 关系过滤、补点逻辑等）。
- 保留 `filterMaxOnePerClassHeatmap(output)`：
  - 限制类（1/2/3/4/7）保留 1 个
  - 其余类最多保留 2 个

### 3.5 结果后处理输出
- 保留坐标回投与缩放逻辑（到热力图坐标系）。
- 保留删除 `id=4` 的逻辑（与原行为一致）。

## 4. 兼容性影响（重要）
- 当前安卓后处理已经是**仅 YOLOv26**方案。
- 如果继续加载旧 YOLOv11 模型，会被按新格式解析，结果会异常。
- 结论：部署时必须使用 YOLOv26 OBB 对应的 NCNN 模型（`.param/.bin`）。

## 5. 安卓调用层是否需要改
- JNI/Java/业务调用接口**不需要改签名**。
- 仍可按原方式调用 `load_obb/run`，输入尺寸继续由调用时传入。
- 只需要切模型文件到新 YOLOv26 OBB 版本。

## 6. 运行日志验收点
- 正常应看到：
  - `流程: YOLOv26_RAW_TOPK`
  - `置信度过滤后候选数: ...`
  - `RAW TopK后候选数(k=300): ...`
  - `类别过滤后: A -> B`
  - `最终输出数量: ...`
- 不应再出现：
  - `LEGACY_RAW`
  - 任何 NMS 相关日志

## 7. 交接后建议执行（由接手同学本地完成）
- 重新编译安卓工程并替换到目标 APK。
- 用同一批热力图样本对比迁移前后：
  - 输出稳定性（规则行为是否符合预期）
  - 性能变化（无 NMS 后通常会略快，最终以实测为准）

## 8. 备注
- 本次未在交接步骤中附带构建命令和运行命令，按项目原有流程执行即可。
