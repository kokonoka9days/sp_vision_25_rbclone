# Model

## 作用

这里保存识别、几何和跟踪阶段共享的单帧观测数据及公共接口。跨帧维护的 `Target` 位于 `tracking/`。

## 主要类型和函数

### `Lightbar` 和 `Armor`

- `Lightbar(rotated_rect, id)`：根据轮廓的最小外接旋转矩形计算灯条中心、方向、长宽和角点。
- `Armor(left, right)`：把两根匹配灯条组合成传统视觉装甲板。
- `Armor(class_id, confidence, box, keypoints)`：把神经网络输出转换成统一装甲板结构。
- `Armor::points`：四个像素角点，是位姿解算的主要输入。
- `Armor::xyz_in_world`、`ypr_in_world`：由 `Solver` 写入的世界坐标和姿态。
- `Armor::name`、`type`、`color`：数字类别、大小装甲和颜色。

### 公共接口

- `IArmorDetector::detect(image)`：所有识别器统一返回 `std::list<Armor>`。
- `IArmorPoseSolver::try_solve(armor)`：位姿求解器把三维结果写回 `Armor`，并用返回值表示成功或失败。

## 原理

一帧图像只能得到离散的装甲板观测，因此 `Armor` 同时保存像素关键点、分类结果和单次三维位姿。识别器写入图像信息，`Solver` 补充空间信息，`Tracker` 再消费同一个结构。

增加字段前要确认识别、几何和跟踪是否真的需要共享它。目标预测、滤波和装甲板关联属于 `tracking/`，不要放回公共数据结构。
