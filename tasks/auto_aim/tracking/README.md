# Tracking

## 作用

把连续帧中的装甲板关联成同一个目标，并维护目标的位置、速度、旋转半径、yaw 和丢失状态。

```text
std::list<Armor> + timestamp -> Tracker -> std::list<Target>
```

## 主要类和函数

### `Tracker`

- `track(armors, t)`：普通自瞄入口；过滤颜色、求解位姿、关联目标并推进状态机。
- `sb_track(armors, t)`：哨兵使用的跟踪入口，包含相应目标选择策略。
- `track(detection_queue, armors, t)`：融合全向感知结果与自瞄装甲板。
- `reset()`：清空当前目标并返回 `lost` 状态。
- `state()`：返回当前状态名称，便于日志和调试。
- `setPoseSolver()`：注入位姿求解接口，测试时可以替换成假实现。
- `set_gimbal()`、`set_fft()`：提供云台状态和周期运动分析器。

内部关键函数：`set_target()` 根据第一帧和目标类型初始化 EKF；`update_target()` 先预测，再选择同名称、同尺寸的装甲板校正；`state_machine()` 管理状态变化。

### `Target`

- `Target(armor, t, radius, armor_num, P0)`：用第一块装甲板初始化目标和 EKF。
- `predict(t)` / `predict(dt)`：按运动模型把状态预测到指定时刻。
- `update(armor)`：用新的装甲板观测校正 EKF。
- `ekf_x()`：读取当前状态向量。
- `armor_xyza_list()`：根据整车中心、半径和 yaw 还原所有装甲板的 `[x,y,z,yaw]`。
- `diverged()`、`convergened()`：判断滤波器是否发散或已经稳定。

`Target` 是跟踪模块对外提供的目标对象；它负责目标语义和生命周期，把具体矩阵计算交给状态估计器。

### 状态估计器

- `RVfromFYT::kf_predict()`：按整车平移、旋转和半径模型预测 11 维状态。
- `prepare_measurement()`：把装甲板位置转换成 `[yaw,pitch,distance,armor_yaw]` 观测。
- `select_armor_id()`：用马氏距离选择观测对应的装甲板编号。
- `correct()`：使用选定装甲板执行 EKF 校正。
- `armor_xyza_list()`：从整车状态还原所有装甲板位置。
- `CAFromTJU::predict_model()`：执行常加速度状态预测。
- `State2Est`：两种状态估计器的公共接口。

`RVfromFYT` 是当前 `Target` 的主要估计器；`CAFromTJU` 是不显式建模装甲板半径的常加速度模型。

### 其他辅助

- `Voter::vote()` / `count()`：累计颜色、编号和装甲板类型出现次数，用于稳定类别判断。
- `enemy_color_from_gimbal()`：把下位机颜色编码转换成视觉颜色。
- `filter_enemy_armors()`：从候选列表中删除非敌方颜色装甲板。

## 原理

跟踪状态依次为 `lost -> detecting -> tracking -> temp_lost`。新目标需要连续检测若干帧才进入稳定跟踪；短暂丢失时继续依靠运动模型预测，超过阈值后才彻底丢弃。匹配成功后，EKF 用新观测校正预测状态。

不同目标使用不同装甲板数量、初始半径和协方差，例如平衡步兵按两块装甲板建模，前哨站和基地按三块建模，普通车辆按四块建模。

EKF 的预测阶段利用运动模型得到下一时刻状态和不确定度；校正阶段将实际观测与预测观测作差，再按协方差权重修正状态。角度残差必须归一化，否则跨越 `-pi/pi` 时会产生错误跳变。

修改丢失和重新锁定逻辑从 `tracker.cpp` 开始；修改目标生命周期从 `target.cpp` 开始；修改状态方程和观测方程从 `rv_from_fyt.cpp` 或 `ca_from_tju.cpp` 开始。
