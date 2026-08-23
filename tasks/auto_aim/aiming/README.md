# Aiming

## 作用

根据跟踪目标、弹速和云台状态，选择未来的击打点，并生成云台及开火命令。

项目中保留了两条控制路线：

```text
Target -> Aimer -> Shooter -> io::Command
Target -> Planner -> Plan(yaw/pitch/速度/加速度/fire)
```

## `Aimer`

- `aim(targets, timestamp, bullet_speed, to_now)`：选择目标，补偿识别延迟和发弹延迟，迭代计算飞行时间，返回 yaw/pitch 命令。
- `aim(..., shoot_mode, ...)`：哨兵左右枪管版本，在普通结果上应用独立 yaw 偏置。
- `choose_aim_point(target)`：从目标的多块装甲板中选择当前应击打的一块。
- `debug_aim_point`：保存最近选择的瞄准点，供重投影和火控检查使用。

原理上，`Aimer` 会先把目标预测到“当前时刻 + 系统延迟”，再计算弹道飞行时间；由于飞行时间依赖目标距离，而目标未来距离又依赖飞行时间，因此最多迭代十次直到相邻结果差小于 1 ms。装甲板选择会限制可见角度，并用锁定编号避免在两块侧向装甲板之间来回跳变。

## `Shooter`

- `shoot(command, aimer, targets, gimbal_pos)`：普通控制命令的开火判断。
- `shoot_g(vision_cmd, aimer, targets, gimbal_pos)`：哨兵通信帧的开火判断。

只有命令有效、存在目标、启用自动开火、瞄准点有效，而且云台实际角度接近上一帧命令时才开火。远近目标可以使用不同角度容差，命令突然跳变时会暂时禁止射击。

## `CommandGener`

- `push(targets, t, bullet_speed, gimbal_pos)`：替换线程中待处理的最新输入。
- `generate_command()`：约 500 Hz 读取最新且不超过 0.2 秒的数据，依次调用 `Aimer`、`Shooter` 和 `CBoard::send()`。

它服务于旧异步流程，只保留最新输入以降低积压延迟。析构函数负责停止并等待工作线程。

## 修改入口

轨迹连续控制和新火控主要看 `planner/`；旧角度命令看 `aimer.cpp`；开火容差看 `shooter.cpp`。弹道、延迟、偏置和阈值由 YAML 控制。
