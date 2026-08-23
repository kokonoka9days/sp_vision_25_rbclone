# Planner

## 作用

把目标未来运动转换成一段期望云台轨迹，再由 MPC 生成当前时刻需要发送的角度、速度和加速度。

## 主要类型和函数

### 输出和入口

- `Plan`：包含 `control`、`fire`、目标 yaw/pitch、控制 yaw/pitch、速度、加速度和 yaw 延迟诊断值。
- `Planner::plan(optional<Target>, bullet_speed, gimbal_yaw, strategy)`：统一入口；先补偿目标模型延迟，再分派具体策略。
- `plan(Target, bullet_speed)`：通用动力学策略。
- `rbplan()`：步兵压制火力策略，包含方向相关 yaw 延迟和前哨站特殊处理。
- `sbplan()`：哨兵策略。
- `rbHeroplan()`：英雄策略。
- `rbShoot()`：根据装甲板位置、角度误差和特殊目标条件判断是否开火。

### 瞄准与轨迹

- `aim()`、`rbaim()`、`heroaim()`：选择装甲板并结合弹道计算期望 yaw/pitch。
- `get_trajectory()`：在规划时间域内不断预测目标，生成 `[yaw,yaw_vel,pitch,pitch_vel]` 参考轨迹。
- `rbget_trajectory_split()`：允许 yaw 和 pitch 使用不同延迟预测目标。
- `setup_yaw_solver()`、`setup_pitch_solver()`：分别建立两轴 TinyMPC，并设置加速度边界与 Q/R 权重。

## 原理

规划时间步长为 `DT=0.01 s`，共 `HORIZON=100` 个点。首先用 `Target::predict()` 和弹道飞行时间得到每个未来时刻的瞄准角，再用中心差分得到角速度，组成参考轨迹。

yaw 和 pitch 分别使用二阶离散模型：状态是角度和角速度，输入是角加速度。MPC 在跟踪误差、控制量大小和最大加速度约束之间求折中。程序取时间域中点作为当前应发送的命令，使规划同时覆盖少量历史和未来轨迹。

开火不是简单比较当前角度，而是比较规划轨迹与可执行轨迹在 `shoot_offset` 位置的误差。`rbplan()` 还会根据命令速度查询 yaw 延迟曲线，并在换向时加入额外惩罚。

## 阅读顺序

先看 `planner.hpp` 中的 `Plan` 和策略枚举，再看 `planner.cpp` 的统一入口，然后看 `planner_trajectory.cpp`。只有需要调整 Q/R、约束或求解器时才进入 `planner_mpc.cpp` 和 `tinympc/`。
