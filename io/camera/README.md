# Camera

## 主要类和函数

### `CameraBase`

- `read(img, timestamp)`：阻塞等待一帧图像。
- `try_read(img, timestamp)`：立即尝试读取，成功时返回 `true`。
- `pause()` / `resume()`：暂停或恢复支持该能力的工业相机。
- `clear_camera_frame_buffer()`：清除 SDK 和软件队列中的旧帧。

所有工业相机实现都继承该接口，并维护采集状态、序列号和最近读取时间。

### `Camera`

- `Camera(config_path)`：根据 YAML 的 `camera_name` 创建大恒、海康或迈德威视相机。
- `read()` / `try_read()`：转发到底层驱动，然后执行统一图像后处理。
- `initSDK()`：初始化需要全局初始化的相机 SDK。
- `pause()`、`resume()`、`is_paused()`：统一控制底层采集状态。

### 图像处理

- `make_protected_gamma_lut(gamma, shadow_offset)`：创建 8 位 Gamma 查找表，并避免暗部被过度拉升。
- `apply_luma_protected_gamma()`：转到 YCrCb，只处理亮度通道，并分别对亮度、色度降噪。

## 原理

具体驱动在后台线程中从 SDK 取帧，立刻记录单调时钟时间戳，再把 `cv::Mat + timestamp` 放入线程安全队列。统一 `Camera` 层不关心厂商句柄，只负责选择实现、修正时间戳并做一致的图像增强。

`USBCamera` 使用 OpenCV `VideoCapture`，是单独的通用相机类，目前不由 `Camera` 工厂创建。

修改采集流程时应保持 `CameraBase` 的时间戳和阻塞语义不变，否则图像与云台姿态可能无法正确对齐。
