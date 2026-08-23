# Geometry

## 作用

根据装甲板四个像素角点、相机标定参数和云台姿态，计算装甲板的三维位置与朝向。

## 主要函数

- `Solver(config_path)`：读取相机内参、畸变、相机到云台外参和 IMU 安装方向。
- `set_R_gimbal2world(q)`：用图像时刻的 IMU 四元数更新云台到世界坐标系的旋转。
- `try_solve(armor)`：执行完整位姿解算，成功后写入 `xyz_in_gimbal`、`xyz_in_world`、`ypr_in_world` 等字段。
- `solve(armor)`：与 `try_solve()` 用途相同，但失败时记录警告。
- `reproject_armor(xyz, yaw, type, name)`：把假设的三维装甲板重新投影成图像角点。
- `world2pixel(points)`：把一组世界坐标点投影到图像。
- `set_camera_calibration()`、`set_camera2gimbal()`：运行时替换相机内外参，主要用于多相机或标定程序。

## 原理

```text
二维角点 + 装甲板真实尺寸
    -> PnP 求相机坐标位姿
    -> 相机外参转换到云台坐标
    -> IMU 姿态转换到世界坐标
    -> 重投影误差优化装甲板 yaw
```

PnP 使用已知三维尺寸和对应像素点求相机相对目标的位姿。由于装甲板正面接近共面，直接得到的 yaw 容易受角点噪声影响，因此 `optimize_yaw()` 会比较实际角点与重投影角点，通过最小化误差进一步修正朝向。

相机内参、畸变、装甲板尺寸和坐标变换全部来自 YAML。角点顺序改变时必须同时检查模型中的三维点顺序。
