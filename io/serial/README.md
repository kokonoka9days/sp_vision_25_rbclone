# Serial

## 主要类和函数

- `serial::Serial(port, baudrate, timeout, ...)`：创建串口对象并设置通信参数。
- `open()`、`close()`、`isOpen()`：管理设备生命周期。
- `read(buffer, size)`、`readline()`：按字节数或行读取数据。
- `write(data)`：发送字节数组、字符串或 vector。
- `available()`、`waitReadable()`：查询或等待可读数据。
- `setPort()`、`setBaudrate()`、`setTimeout()`：修改端口、波特率和超时。
- `flushInput()`、`flushOutput()`：清理系统串口缓冲区。
- `serial::list_ports()`：枚举系统串口及硬件 ID。

## 原理

这是一个跨平台串口库，公共 API 位于 `include/serial/serial.h`，Linux、macOS 和 Windows 的系统调用实现在 `src/impl/`。项目中的 `Gimbal` 和 `DM_IMU` 在其上实现自己的定长二进制协议。

除非修改串口兼容性、超时或底层读写行为，否则不需要从视觉算法入口阅读这里。
