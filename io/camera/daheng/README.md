# Daheng Camera

## 主要函数

- `DahengCamera::initSDK()`：初始化大恒 Galaxy SDK。
- `enum_and_check_camera()`：枚举设备并按序列号确认目标相机存在。
- `open_camera()` / `initialize_camera()`：打开设备并配置曝光、增益、Gamma、像素格式等参数。
- `getFrame()`：从 SDK 获取一帧原始数据。
- `ProcessData()`：完成 Bayer/Mono 转换、翻转、镜像和 OpenCV 图像构造。
- `read()` / `try_read()`：从软件队列读取带时间戳图像。
- `pause()` / `resume()`：控制采集线程暂停和恢复。

## 原理

采集线程不断从 Galaxy SDK 拉取原始帧，按像素格式转换为 BGR，并将图像及采集时刻放入队列。守护线程负责发现采集异常后重新打开设备。`include/` 是厂商 SDK 头文件，不属于项目算法代码。
