# MindVision Camera

## 主要函数

- `MindVision(exposure_us, gamma, vid_pid)`：保存参数并启动设备与采集管理。
- `open()`：枚举相机、初始化 SDK、设置曝光/Gamma 并开始采集。
- `try_open()`：捕获打开异常，供守护线程持续重试。
- `read()` / `try_read()`：从队列读取图像和采集时间戳。
- `close()`：停止设备并释放 SDK 句柄。
- `reset_usb()`：按 VID/PID 复位异常 USB 设备。

## 原理

采集线程从迈德威视 SDK 获取帧并压入队列；连接异常时守护线程关闭旧句柄并重新打开。`include/` 和 `lib/` 是厂商提供的接口及不同架构预编译库。
