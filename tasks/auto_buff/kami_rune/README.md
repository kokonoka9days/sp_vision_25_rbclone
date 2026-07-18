# 已激活扇叶灯条骨架提取

## 1. 项目目标

这个小项目负责从已激活扇叶的二值轮廓中恢复灯条骨架。

算法不直接使用多边形角点描述轮廓，而是把闭合轮廓转换为一维方向信号，从中找出方向稳定的直线边界。由于同一根灯条的两侧边界在轮廓中遍历方向相反，因此可以通过正反边界配对恢复位于二者中间的灯条骨架。

骨架结果包含起点、终点、中心、方向和长度，同时保留边界间距、方向误差、投影重叠率和匹配分数，便于后续筛选、角点求交和调试。

## 2. 整体框架

```text
CV_8UC1 二值图
       |
       v
CHAIN_APPROX_NONE 提取完整闭合轮廓
       |
       v
面积、外接矩形、填充率、紧致度候选筛选
       |
       v
相邻轮廓点差分 -> 闭合高斯滤波 -> 方向角和角度梯度
       |
       v
低梯度连续区间 -> 短缺口合并 -> fitLine 边界拟合
       |
       v
方向相差约 180 度的边界候选
       |
       v
轮廓间隔、灯条宽度、投影重叠、纵向偏移检查
       |
       v
候选评分和全局一对一匹配
       |
       v
灯条中心线骨架
```

项目同时提供两个入口：

- `extract()`：从二值图开始，执行轮廓提取、候选筛选和骨架恢复。
- `analyze_contour()`：直接分析已经由其他模块筛选好的单个闭合轮廓。

第二种入口适合接入现有 `ActiveFanDetector`，避免重复执行轮廓提取和候选筛选。

此外，`detect_and_show_gradient_endpoints()` 提供独立的轮廓端点入口。它对有符号
方向梯度再次执行圆周高斯低通，将相邻且空间接近的凸响应合并为一个凸起区域，
并保证每个区域只输出一个端点。普通凸起取区域主峰；粗灯条的平头若形成两个
近似直角峰，则在确认两峰之间存在明显低谷后取两峰中点。该流程不依赖骨架配对，
也不使用跨帧跟踪。所有通过过滤的凸起区域端点都会被返回和绘制。

## 3. 文件结构

```text
kami_rune/
|-- fan_skeleton_extractor.hpp       公共接口、参数和结果数据结构
|-- fan_skeleton_extractor.cpp       方向信号、边界拟合、配对及骨架恢复实现
|-- fan_skeleton_extractor_test.cpp  合成轮廓测试
|-- CMakeLists.txt                   独立构建和测试入口
`-- README.md                        项目说明
```

当前目录可以独立编译，没有修改或依赖父级 `auto_buff/CMakeLists.txt`。

## 4. 核心数据结构

### `FanSkeletonParams`

保存完整算法参数，主要分为四组：

1. 轮廓候选参数：面积、长宽比、填充率和紧致度。
2. 方向信号参数：高斯窗口长度、窗口比例和标准差。
3. 边界提取参数：最大方向梯度、可合并缺口和直线拟合误差。
4. 正反边界配对参数：反向角容差、灯条宽度、重叠率、长宽比和纵向偏移。

所有参数都具有默认值，也可以在构造 `FanSkeletonExtractor` 前单独调整。

### `DirectionSignal`

保存轮廓信号化后的中间结果：

- `raw_directions`：相邻轮廓点的原始单位方向。
- `smoothed_directions`：经过闭合高斯滤波的方向。
- `angles_deg`：平滑方向对应的角度。
- `gradient_deg`：方向角沿轮廓的局部变化率。

方向梯度通过向量点积和叉积计算，不依赖展开后的角度数值，因此不会受到 `-180/180` 度跳变影响。

### `BoundarySegment`

表示一段已经压缩和拟合的直线边界，包含：

- 原轮廓中的起止下标和采样数量。
- 是否跨越闭合轮廓末尾。
- 拟合中心、方向和两个端点。
- 边界长度和直线拟合均方根误差。

### `BoundaryPair`

表示一组成功匹配的正反边界，包含：

- 两条边界在 `boundaries` 中的下标。
- 恢复出的 `SkeletonLine`。
- 灯条宽度。
- 正反方向误差。
- 轴向投影重叠率。
- 综合匹配分数，数值越小表示匹配质量越好。

### `ContourSkeletonResult`

保存一个轮廓的完整处理结果：原轮廓、方向信号、拟合边界和最终边界对。中间数据全部保留，方便绘图、参数调整和问题定位。

## 5. 核心处理流程

### 5.1 完整轮廓与候选筛选

`extract()` 要求输入非空的 `CV_8UC1` 二值图，并通过 `CHAIN_APPROX_NONE` 保留所有轮廓像素。候选筛选用于排除面积异常、过于狭长、填充率异常或紧致度异常的轮廓。

如果外部模块已经完成筛选，可以直接调用 `analyze_contour()` 跳过这一步。

### 5.2 闭合方向信号

对于轮廓点 `P[i]`，首先计算：

```text
D[i] = normalize(P[i + 1] - P[i])
```

下标按照轮廓长度循环，因此最后一个点会自然连接第一个点。之后对方向向量进行圆周高斯滤波，再计算每个位置的角度和局部角度梯度。

这种实现直接把轮廓视为周期数据，不需要在尾部复制一段轮廓，也不会因为轮廓起点变化而改变结果。

### 5.3 直线边界提取

方向梯度较小的位置被标记为稳定区域。短暂的不稳定缺口只有在长度足够小且两端方向一致时才会被填补，避免简单形态学闭操作错误连接不同方向的边界。

每个连续稳定区间使用 `cv::fitLine` 拟合为 `BoundarySegment`。拟合方向会按照原轮廓遍历方向进行统一，并根据长度和均方根误差删除质量较差的边界。

### 5.4 正反边界配对和骨架恢复

同一灯条的两侧边界应满足以下条件：

- 遍历方向接近相反，即方向差接近 180 度。
- 在闭合轮廓上的间隔不能过大。
- 两条边界的法向距离符合灯条宽度范围。
- 边界长度与宽度之比符合条状结构。
- 沿骨架方向具有足够的投影重叠。
- 两条边界中心的纵向错位不能过大。

通过检查的候选会计算综合分数。候选按分数从小到大排序，每条边界最多参与一次匹配，从而得到确定的一对一边界组合。

骨架位于两条边界的法向中间位置，起止范围取两条边界沿轴向投影的公共区间。这可以避免将只有少量重叠的边界外推成过长骨架。

## 6. 使用示例

### 从二值图提取

```cpp
#include "tasks/auto_buff/kami_rune/fan_skeleton_extractor.hpp"

auto_buff::kami_rune::FanSkeletonExtractor extractor;
std::vector<auto_buff::kami_rune::ContourSkeletonResult> results =
  extractor.extract(binary);

for (const auto & result : results) {
  for (const auto & pair : result.pairs) {
    const auto & skeleton = pair.skeleton;
    cv::line(image, skeleton.start, skeleton.end, cv::Scalar(0, 255, 0), 2);
  }
}
```

### 分析已有轮廓

```cpp
auto_buff::kami_rune::FanSkeletonParams params;
params.opposite_angle_tolerance_deg = 15.0;
params.min_pair_overlap_ratio = 0.4;

auto_buff::kami_rune::FanSkeletonExtractor extractor(params);
auto result = extractor.analyze_contour(selected_contour, contour_id);
```

## 7. 构建与测试

在 `kami_rune` 目录下执行：

```bash
cmake -S . -B /tmp/kami_rune_build
cmake --build /tmp/kami_rune_build --parallel
ctest --test-dir /tmp/kami_rune_build --output-on-failure
```

当前测试覆盖：

- 旋转矩形灯条的方向、中心和长度恢复。
- 改变闭合轮廓起点后结果保持一致。
- 从二值图开始的完整处理入口。
- 三根相连平行灯条的多骨架恢复和一对一配对。

## 8. 与主工程的关系

本目录的实现已经写入父级 `auto_buff` 构建目标。上层 `ActiveFanDetector` 负责颜色
二值化、big ROI、rune_center 轮廓面积下限和深度检测框过滤，再将
`remaining_contours` 传入端点提取接口。

骨架恢复之后，上层还可以继续实现：

- 根据多条骨架的空间关系识别顶部、侧面和底部灯条。
- 计算骨架交点并生成扇叶特征角点。
- 按神符中心方向筛选错误骨架。
- 将二维角点映射到扇叶三维模型并参与 PnP。

真实相机数据中的目标尺度、曝光和轮廓毛刺会影响默认参数，接入主工程后应结合实际视频调整候选阈值、平滑窗口、直线梯度阈值和配对宽度范围。
