# Gimbal

## 当前串口接口 `Gimbal`

- `state()`：在线程锁保护下复制最新 yaw、pitch、弹速、弹丸计数和敌方颜色。
- `mode()`：返回空闲、自瞄、小符、大符或长焦模式。
- `q(t)`：从姿态队列取出图像时间前后的四元数，用球面线性插值得到图像时刻姿态。
- `send(control, fire, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc)`：组装普通自瞄帧并发送。
- `sb_send()`：发送附带目标位置和类别的哨兵帧。
- `omni_send()`：发送全向感知的 yaw、pitch 和距离。
- `open_serial()`、`reconnect()`：在多个候选串口中尝试连接，并在掉线后恢复。

`GimbalToVision`、`VisionToGimbal` 等 packed 结构直接定义线上的字节布局。接收线程校验帧头和 CRC，解析状态并把姿态与接收时间放入队列。发送帧使用局部变量组装，避免自瞄和打符线程同时写共享缓冲区。

## 旧 CAN 接口 `CBoard`

- `imu_at(timestamp)`：在相邻 CAN 四元数之间做 SLERP 插值。
- `send(Command)`：把控制、开火、yaw、pitch 和距离编码到 8 字节 CAN 帧。
- `callback(frame)`：按 CAN ID 解析四元数、弹速、模式和射击侧。
- `SocketCAN::write()`：发送 Linux CAN 帧；内部用 epoll 接收并调用回调。

## 原理和修改注意

相机曝光与姿态接收不是同一时刻，因此必须通过时间戳插值获得正确姿态。四元数使用 SLERP，避免直接对欧拉角线性插值造成跳变。

修改协议时必须同步核对下位机结构体、字段单位、大小端、packed 大小、帧头、帧尾和 CRC。当前 `VisionToGimbal::crc16` 发送路径仍固定为 0，启用下行校验前需要和电控确认。
