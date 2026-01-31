# 文件路径: sp_msgs/msg/EnemyStatusMsg.msg

# 使用标准接口的时间戳，包含 sec 和 nanosec
builtin_interfaces/Time timestamp

# 对应代码中的 std::vector<int8_t>，用于存储无敌状态的敌人ID
int8[] invincible_enemy_ids