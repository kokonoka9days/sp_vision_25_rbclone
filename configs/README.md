# 配置文件参数参考

本文档依据当前代码中的 YAML 读取逻辑整理。项目没有一个适用于所有程序的单一配置结构；每个可执行程序会按其创建的模块读取同一个 YAML 文件中的不同字段。

## 字段状态说明

| 状态 | 含义 |
| --- | --- |
| 必填 | 对应模块一旦创建就会读取；缺失时程序退出或抛出 YAML 异常 |
| 条件必填 | 只在选择对应相机、推理后端或工具时必填 |
| 可选 | 可以不写，代码中有明确默认值 |
| 工具必填 | 仅特定标定或测试工具读取 |

相对路径按程序启动时的当前工作目录解析，并非按 YAML 文件所在目录解析。数组长度必须符合表中要求；当前求解代码直接按固定长度映射，长度错误可能导致越界访问。

## 工业相机

`io::Camera` 支持 `daheng`、`hikrobot` 和 `mindvision`。公共字段会在判断相机厂商前读取。

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `camera_name` | string | 必填 | 相机实现，可选 `daheng`、`hikrobot`、`mindvision` |
| `exposure_us` | double | 必填 | 曝光时间，单位 us |
| `timestamp_offset_us` | int | 必填 | 采集时间戳补偿量，单位 us；保存到 `Camera::timestamp_offset`，由上层按需使用，`Camera::read` 本身不自动修正时间戳 |
| `flip` | bool | 必填 | 是否垂直翻转；大恒和海康使用，迈德威视当前不使用，但公共构造流程仍要求字段存在 |
| `mirror` | bool | 必填 | 是否水平镜像；大恒和海康使用，迈德威视当前不使用，但公共构造流程仍要求字段存在 |
| `img_gamma` | double | 必填 | 软件亮度 Gamma；必须大于 0，`1.0` 表示完全跳过软件 Gamma 和配套降噪 |
| `img_gamma_shadow_offset` | double | 可选，`0.04` | 暗部保护偏移，范围 `[0,1]`；越大越限制低亮度区域的 Gamma 增益 |
| `img_gamma_luma_denoise_sigma` | double | 可选，`0.7` | Gamma 前 Y 亮度通道 3x3 高斯降噪 sigma；`0` 关闭 |
| `img_gamma_chroma_denoise_sigma` | double | 可选，`1.0` | Gamma 前 Cr/Cb 色度通道 3x3 高斯降噪 sigma；`0` 关闭 |
| `camera_sn` | string | 大恒/海康条件必填 | 用于选择指定序列号的工业相机 |
| `gain` | double | 大恒/海康条件必填 | 归一化增益比例，预期 `[0,1]`；`1` 对应相机增益范围上限，容易显著增加噪声 |
| `gamma` | double | 大恒/迈德威视条件必填 | 相机或 SDK ISP Gamma；海康分支不读取 |
| `vid_pid` | string | 迈德威视条件必填 | USB VID:PID，十六进制格式，例如 `f622:d13a`；用于掉线后的 USB 复位 |

海康的 VID:PID 当前在代码中固定为 `2bdf:0001`，大恒固定为 `2ba2:4d55`，因此这两个分支不会读取 YAML 中的 `vid_pid`。

## USB 相机

以下字段仅在创建 `io::USBCamera` 时必填，值最终通过 OpenCV V4L2 属性设置，实际有效范围由摄像头驱动决定。

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `image_width` | double | 必填 | 请求的图像宽度，单位 px |
| `image_height` | double | 必填 | 请求的图像高度，单位 px |
| `usb_frame_rate` | double | 必填 | 请求帧率，单位 FPS |
| `usb_exposure` | double | 必填 | `CAP_PROP_EXPOSURE` 的驱动值，不保证等同于 us |
| `usb_gamma` | double | 必填 | `CAP_PROP_GAMMA` 的驱动值 |
| `usb_gain` | double | 必填 | `CAP_PROP_GAIN` 的驱动值 |

USB 相机设备名不在 YAML 中读取，而是通过 `USBCamera(open_name, config_path)` 的 `open_name` 参数传入，例如 `video0`。

## 串口云台与 CAN 板

### Gimbal

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `com_port` | string 或 string[] | 必填 | 串口路径或候选路径列表；按顺序尝试，波特率固定为 460800 |
| `gimbal_y1` | int | 必填 | 将电控姿态轴映射到视觉 yaw；符号表示是否取反 |
| `gimbal_p2` | int | 必填 | 将电控姿态轴映射到视觉 pitch；符号表示是否取反 |
| `gimbal_r3` | int | 必填 | 将电控姿态轴映射到视觉 roll；符号表示是否取反 |

轴映射通常使用绝对值 `1/2/3` 表示三个输入轴，例如 `gimbal_r3: -3` 表示第三轴取反。

### CBoard

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `quaternion_canid` | int | 必填 | 接收 IMU 四元数的 CAN ID，可写十六进制 |
| `bullet_speed_canid` | int | 必填 | 接收弹速等状态的 CAN ID |
| `send_canid` | int | 必填 | 发送视觉控制数据的 CAN ID |
| `can_interface` | string | 必填 | SocketCAN 接口名，例如 `can0` |

## 自瞄推理后端

### 后端选择

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `yolo_name` | string | 必填 | OpenVINO 构建支持 `yolov5`、`ov_0526`、`yolov8`、`yolo11`；TensorRT 构建支持 `trt_0526`、`trt_0708` |
| `device` | string | OpenVINO 必填；TRT 0708 可选，`cuda:0` | OpenVINO YOLOv5 支持 `CPU`、`GPU`、`GPU.n`、`AUTO`；YOLOv8/YOLO11 直接交给 OpenVINO；TRT 0708 作为 CUDA 设备标识 |
| `min_confidence` | double | OpenVINO 必填；TRT 0526 默认 `0.65`；TRT 0708 默认 `0.5` | 最终候选置信度下限，通常取 `[0,1]` |
| `detect_color` | int | TRT 可选，`-1` | TensorRT 颜色过滤：`-1` 全部、`0` 红、`1` 蓝 |

### 模型路径

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `yolov5_model_path` | string | `yolo_name: yolov5` 时必填 | OpenVINO YOLOv5 模型路径 |
| `ov_0526_model_path` | string | `yolo_name: ov_0526` 时必填 | OpenVINO 0526 模型路径 |
| `yolov8_model_path` | string | `yolo_name: yolov8` 时必填 | OpenVINO YOLOv8 模型路径 |
| `yolo11_model_path` | string | `yolo_name: yolo11` 时必填 | OpenVINO YOLO11 模型路径 |
| `trt_engine_path_0526` | string | `yolo_name: trt_0526` 时必填 | 0526 TensorRT engine 路径 |
| `trt_engine_path_0708` | string | `yolo_name: trt_0708` 时必填 | 0708 TensorRT engine 路径 |
| `classify_model` | string | OpenVINO YOLO/传统检测条件必填 | 装甲板数字分类 ONNX 路径；OpenVINO YOLO 类会构造传统检测器，因此即使 `use_traditional: false` 仍需提供 |

### ROI 和 YOLO 阈值

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `use_roi` | bool | OpenVINO 必填；TRT 0708 默认 `false` | 是否只在 ROI 内推理；TRT 0526 不读取 |
| `roi.x` | int | OpenVINO 必填；TRT 0708 默认 `0` | ROI 左上角 x，单位 px |
| `roi.y` | int | OpenVINO 必填；TRT 0708 默认 `0` | ROI 左上角 y，单位 px |
| `roi.width` | int | OpenVINO 必填；TRT 0708 默认 `-1` | ROI 宽度；`-1` 表示延伸到图像右边界 |
| `roi.height` | int | OpenVINO 必填；TRT 0708 默认 `-1` | ROI 高度；`-1` 表示延伸到图像下边界 |
| `use_traditional` | bool | YOLOv5/OV 0526 必填 | 是否用传统检测结果复核 YOLO 装甲板；其他后端不读取 |
| `yolo_score_threshold` | float | YOLOv5 默认 `0.7`；OV 0526 默认 `0.65` | 网络候选分数阈值 |
| `yolo_nms_threshold` | float | YOLOv5 默认 `0.3`；OV 0526 默认 `0.45` | NMS IoU 阈值 |

OpenVINO 实现会在读取 `use_roi` 前无条件读取 `roi` 四个子字段，因此即使 `use_roi: false`，ROI 节点也必须完整存在。

## 传统装甲板检测与数字分类

以下几何参数由 `auto_aim::Detector` 和 `rv_aim::Detector` 读取。OpenVINO YOLO 会构造 `auto_aim::Detector`，因此 OpenVINO 配置通常也必须包含这些字段。

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `threshold` | double | 必填 | 灰度二值化阈值；YOLO11/YOLOv8 也读取该值，但其成员目前没有额外用途 |
| `max_angle_error` | double | 必填 | 成对灯条最大角度差，单位 degree |
| `min_lightbar_ratio` | double | 必填 | 灯条长宽比下限 |
| `max_lightbar_ratio` | double | 必填 | 灯条长宽比上限 |
| `min_lightbar_length` | double | 必填 | 灯条最小长度，单位 px |
| `min_armor_ratio` | double | 必填 | 装甲板宽高比下限 |
| `max_armor_ratio` | double | 必填 | 装甲板宽高比上限 |
| `max_side_ratio` | double | 必填 | 左右灯条长度比允许上限 |
| `max_rectangular_error` | double | 必填 | 装甲板矩形几何误差上限，单位 degree |

`min_confidence` 同时被传统检测器作为数字分类置信度下限使用，见推理后端表。

RV 数字分类器额外读取：

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `number_classifier.model_path` | string | RV 条件必填 | RV 数字分类模型路径 |
| `number_classifier.label_path` | string | RV 条件必填 | 类别标签文件路径 |
| `number_classifier.threshold` | double | 可选，`0.5` | RV 数字分类置信度阈值 |

## 位姿求解与坐标标定

以下字段由自瞄和能量机关 Solver 读取，均为行优先数组。

| 字段 | 类型/长度 | 状态 | 作用 |
| --- | --- | --- | --- |
| `R_gimbal2imubody` | double[9] | Solver 必填 | 云台坐标系到 IMU 机体坐标系的 3x3 旋转矩阵 |
| `R_camera2gimbal` | double[9] | Solver 必填 | 相机坐标系到云台坐标系的 3x3 旋转矩阵 |
| `t_camera2gimbal` | double[3] | Solver 必填 | 相机原点在云台坐标系中的平移，单位 m |
| `camera_matrix` | double[9] | Solver 必填 | OpenCV 3x3 相机内参矩阵 |
| `distort_coeffs` | double[5] | Solver 必填 | OpenCV 畸变系数 `[k1,k2,p1,p2,k3]` |

## Tracker

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `enemy_color` | string | 必填 | 敌方颜色；`red` 选择红方，其余值按蓝方处理。全向感知额外识别 `auto` 并从云台状态获取颜色 |
| `min_detect_count` | int | 必填 | 连续检测达到该帧数后，从 detecting 进入 tracking |
| `max_temp_lost_count` | int | 必填 | 普通目标临时丢失的最大容忍帧数 |
| `outpost_max_temp_lost_count` | int | 必填 | 前哨站目标临时丢失的最大容忍帧数 |

## Aimer 与 Shooter

`auto_aim::Aimer` 是旧瞄准接口；当前 MPC `Planner` 也复用其中若干字段。

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `yaw_offset` | double | Aimer/Planner/Buff Aimer 必填 | yaw 静态补偿，单位 degree |
| `pitch_offset` | double | Aimer/Planner/Buff Aimer 必填 | pitch 静态补偿，单位 degree |
| `comming_angle` | double | Aimer 必填 | 旋转目标进入可瞄区的角度阈值，单位 degree；字段名按代码保留为 `comming` |
| `leaving_angle` | double | Aimer 必填 | 旋转目标离开可瞄区的角度阈值，单位 degree |
| `decision_speed` | double | Aimer/Planner 必填 | 高低速预测策略分界角速度，单位 rad/s |
| `high_speed_delay_time` | double | Aimer/Planner 必填 | 高速目标额外预测时间，单位 s |
| `low_speed_delay_time` | double | Aimer/Planner 必填 | 低速目标额外预测时间，单位 s |
| `left_yaw_offset` | double | 可选，但必须与 `right_yaw_offset` 同时出现 | 左侧射击模式 yaw 补偿，单位 degree |
| `right_yaw_offset` | double | 可选，但必须与 `left_yaw_offset` 同时出现 | 右侧射击模式 yaw 补偿，单位 degree |
| `first_tolerance` | double | Shooter 必填 | 近距离射击角度容差，单位 degree |
| `second_tolerance` | double | Shooter 必填 | 远距离射击角度容差，单位 degree |
| `judge_distance` | double | Shooter 必填 | 近距离与远距离的分界，单位 m |
| `auto_fire` | bool | Shooter 必填 | 是否允许 Shooter 自动下发开火 |

## MPC Planner

下列字段在创建 `auto_aim::Planner` 时全部必填，没有代码默认值。

| 字段 | 类型 | 单位/作用 |
| --- | --- | --- |
| `far_pitch_offset` | double | 远距离 pitch 补偿，degree |
| `far_high_pitch_offset` | double | 同时满足远距离和高目标条件时的 pitch 补偿，degree |
| `target_dist_error` | double | 目标水平距离补偿，m |
| `target_h_error` | double | 目标高度补偿，m |
| `fire_thresh` | double | MPC 预测轨迹与控制轨迹的角度残差开火阈值，rad |
| `small_armor_tolerance` | double | 小装甲板开火角窗口，rad |
| `big_armor_tolerance` | double | 大装甲板开火角窗口，rad |
| `tower_and_base_armor_tolerance_` | double | 前哨站和基地装甲板开火角窗口，rad；字段末尾下划线是配置名的一部分 |
| `gimbal_control_delay` | double | 云台控制延迟补偿，s |
| `tower_pitch_prediction_time` | double | 前哨站 pitch 预测时间，s |
| `gimbal_delay` | double | 加到高/低速预测时间上的云台总延迟，s |
| `shoot_offset` | int | MPC 中心点后的开火检查步偏移；当前合法范围 `[-50,49]` |
| `max_yaw_acc` | double | yaw MPC 控制量上限，rad/s^2 |
| `Q_yaw` | double[2] | yaw 状态 `[角度,角速度]` 代价权重 |
| `R_yaw` | double[1] | yaw 控制量代价权重 |
| `max_pitch_acc` | double | pitch MPC 控制量上限，rad/s^2 |
| `Q_pitch` | double[2] | pitch 状态 `[角度,角速度]` 代价权重 |
| `R_pitch` | double[1] | pitch 控制量代价权重 |

## 全向感知

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `main_and_secondary` | string | 全向双相机入口条件必填 | 相机方位标识；当前有效逻辑使用 `left` 和 `right` |
| `image_width` | double | Decider 必填 | 配置图像宽度；当前 Decider 读取后未参与计算 |
| `image_height` | double | Decider 必填 | 配置图像高度；当前 Decider 读取后未参与计算 |
| `fov_h` | double | Decider 必填 | 原水平视场角，degree；当前读取后未参与计算 |
| `fov_v` | double | Decider 必填 | 原垂直视场角，degree；当前读取后未参与计算 |
| `new_fov_h` | double | Decider 必填 | 用于归一化像素坐标换算 yaw 的有效水平视场角，degree |
| `new_fov_v` | double | Decider 必填 | 用于归一化像素坐标换算 pitch 的有效垂直视场角，degree |
| `mode` | int | Decider 必填 | 目标优先级模式，当前有效值为 `1`、`2`、`3` |

全向感知还会读取前述 `enemy_color`；值为 `auto` 时从云台数据动态选择敌方颜色。

## 能量机关

### 推理模型

| 字段 | 类型 | 状态/默认值 | 作用 |
| --- | --- | --- | --- |
| `model` | string | OpenVINO Buff 条件必填 | 能量机关 YOLO11 OpenVINO/ONNX 模型路径 |
| `buff_engine_path` | string | TensorRT Buff 条件必填 | 能量机关 TensorRT engine 路径 |
| `buff_confidence_threshold` | float | 可选，`0.7` | 候选框置信度阈值 |
| `buff_keypoint_threshold` | float | 可选，`0.3` | 关键点置信度阈值，同时用于新建轨迹 |
| `buff_iou_threshold` | float | 可选，`0.4` | NMS IoU 阈值 |

### 运动模型与拟合

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `buff_rune_radius_m` | double | `0.700` | R 标中心到目标运动圆周的半径，m |
| `buff_small_direction` | int | `0` | 小符旋转方向；`0` 自动判断，正数强制 `+1`，负数强制 `-1` |
| `buff_blind_timeout_s` | double | `0.100` | 丢失观测后仍允许盲控的时间，s |
| `buff_track_retention_s` | double | `0.400` | 未观测轨迹的保留时间，s |
| `buff_fire_full_observation_max_age_s` | double | `0.030` | 开火所用完整观测允许的最大年龄，s |
| `buff_direction_confirm_intervals` | int | `3` | 自动旋转方向连续确认的时间间隔数量 |
| `buff_big_speed_phase_window` | int | `7` | 大符角速度估计保留的相位样本数 |
| `buff_big_speed_min_span_s` | double | `0.030` | 大符速度估计要求的最小样本时间跨度，s |
| `buff_big_fit_min_span_s` | double | `1.0` | 大符正弦速度拟合要求的最小时间跨度，s |
| `buff_big_fit_min_inlier_ratio` | double | `0.75` | 大符拟合最小内点比例 |
| `buff_big_fit_max_rms` | double | `0.18` | 大符拟合允许的最大 RMS 残差 |
| `buff_big_fit_blend_s` | double | `0.30` | 大符拟合结果渐入预测器的混合时间，s |

### 检测、中心与轨迹关联

以下字段全部可选。

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `buff_keypoint_hard_threshold` | `0.15` | 保留弱关键点的最低硬阈值 |
| `buff_keypoint_temporal_gate_px` | `10.0` | 关键点时序残差门限，px |
| `buff_center_innovation_gate_px` | `45.0` | R 标中心更新创新门限，px |
| `buff_center_recovery_hits` | `2` | 中心恢复所需连续命中数 |
| `buff_center_lost_max` | `6` | 中心允许连续丢失的最大帧数 |
| `buff_pair_angle_gate_deg` | `15` | 目标与扇叶配对的角度门限，degree |
| `buff_pair_ratio_min` | `0.30` | 配对半径比例下限 |
| `buff_pair_ratio_max` | `0.70` | 配对半径比例上限 |
| `buff_pair_ratio_center` | `0.51` | 配对半径比例期望中心 |
| `buff_track_gate_min_deg` | `12` | 轨迹关联角门限的最小值，degree |
| `buff_track_gate_max_deg` | `25` | 轨迹关联角门限的最大值，degree |
| `buff_track_reset_timeout_s` | `0.500` | 超时后重置轨迹状态，s |
| `buff_track_confirm_hits` | `2` | 新轨迹确认所需连续命中数 |
| `buff_track_recovery_hits` | `2` | 丢失轨迹恢复所需连续命中数 |
| `buff_control_blind_timeout_s` | `buff_blind_timeout_s` | 控制输出允许的盲控时间，s |
| `buff_switch_confirm_frames` | `5` | 通用目标切换确认帧数；显式配置时也会先覆盖相邻槽切换帧数 |
| `buff_same_slot_confirm_frames` | `3` | 同槽目标切换确认帧数 |
| `buff_adjacent_switch_confirm_frames` | `8` | 相邻槽目标切换确认帧数；若同时配置，优先于 `buff_switch_confirm_frames` |
| `buff_adjacent_switch_delay_s` | `0.180` | 相邻槽切换后的等待时间，s |
| `buff_slot_tolerance_deg` | `12` | 将观测归入扇叶槽位的角容差，degree |
| `buff_switch_pair_angle_gate_deg` | `10` | 切换候选的配对角门限，degree |
| `buff_switch_pair_ratio_min` | `0.38` | 切换候选半径比例下限 |
| `buff_switch_pair_ratio_max` | `0.64` | 切换候选半径比例上限 |

### PnP 门限

以下字段全部可选；能量机关 Solver 仍要求位姿求解章节中的五个标定字段。

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `buff_pnp_full_reprojection_gate_px` | `6.0` | 完整 8 点 PnP 最大重投影误差，px |
| `buff_pnp_target_center_gate_px` | `6.0` | 仅目标观测的中心重投影门限，px |
| `buff_pnp_fan_center_gate_px` | `8.0` | 仅扇叶观测的中心重投影门限，px |
| `buff_pnp_partial_center_gate_px` | `8.0` | 部分 4 点 PnP 中心门限，px |
| `buff_pnp_partial_angle_gate_deg` | `15` | 部分 4 点 PnP 角度门限，degree |

### Buff Aimer

| 字段 | 类型 | 状态 | 作用 |
| --- | --- | --- | --- |
| `yaw_offset` | double | 必填 | 能量机关 yaw 补偿，degree |
| `pitch_offset` | double | 必填 | 能量机关 pitch 补偿，degree |
| `fire_gap_time` | double | 必填 | 两次开火的最小时间间隔，s |
| `predict_time` | double | 必填 | 能量机关额外前向预测时间，s |

## 标定工具

| 字段 | 类型/长度 | 状态 | 读取者与作用 |
| --- | --- | --- | --- |
| `pattern_cols` | int | 工具必填 | 棋盘格每行内角点数；`capture`、相机标定和联合标定读取 |
| `pattern_rows` | int | 工具必填 | 棋盘格每列内角点数 |
| `square_size_mm` | double | 工具必填 | 棋盘格方格边长，mm |
| `R_gimbal2imubody` | double[9] | 手眼/联合标定必填 | 标定时的云台到 IMU 机体旋转 |
| `camera_matrix` | double[9] | 手眼标定必填或联合标定输出 | 相机内参 |
| `distort_coeffs` | double[5] | 手眼标定必填或联合标定输出 | 相机畸变参数 |
| `R_camera2gimbal` | double[9] | 联合标定输出/校验 | 相机到云台旋转 |
| `t_camera2gimbal` | double[3] | 联合标定输出/校验 | 相机到云台平移，m |

`calibrate_input` 会在原 YAML 中更新或新增 `camera_matrix`、`distort_coeffs`、`R_camera2gimbal` 和 `t_camera2gimbal`，并校验上述固定长度。

## 测试工具专用字段

`tests/sp_example/handeye_test.cpp` 额外读取以下字段，均无默认值：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `height` | double | 投影测试网格相对世界原点的高度 |
| `grid_num` | int | 每个方向生成的网格数量 |
| `grid_size` | double | 网格点间距，使用求解器世界坐标单位 |
| `delay` | int | 查询历史 IMU 姿态的延迟，单位 ms |

`auto_buff_test` 在命令行未指定调试预测时间时可选读取 `predict_time`；缺失时按 `0` 处理，并额外加固定 `0.1 s`。

## 当前样例中存在但生产代码不读取的字段

以下字段出现在一个或多个 `configs/*.yaml` 中，但当前生产路径没有对应读取逻辑。修改它们不会改变当前运行行为：

- `tensorrt_engine_path`：旧的通用 TensorRT 路径；当前使用 `trt_engine_path_0526` 或 `trt_engine_path_0708`。
- `detect.*`：旧能量机关传统图像处理参数；当前代码没有读取这些字段。
- `buff_locked_gate_deg`、`buff_switch_gate_deg`、`buff_locked_lost_max`：旧 Buff 跟踪字段。
- `aim_time`、`wait_time`、`command_fire_gap`：旧 Buff 瞄准/发射字段。
- `number_classifier.ignore_classes`：RV 分类器当前不读取。
- `new_image_width`、`new_image_height`、`new_usb_exposure`：当前 USB 相机和 Decider 不读取。
- `min_spin_speed`：当前 Aimer 不读取。
- `yaw_kp`、`yaw_kd`、`pitch_kp`、`pitch_kd`：当前控制代码不从 YAML 读取。
- 顶层 `CAMERA`：当前 YAML 读取代码不使用该节点。

是否必填最终取决于可执行程序实际创建的模块。例如，只运行相机采集测试不需要 Tracker 和 Planner 参数；运行完整自瞄程序时，相机、Gimbal、YOLO、Detector、Solver、Tracker 和 Planner 所需字段通常必须同时存在。
