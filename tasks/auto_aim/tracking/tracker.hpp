#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <optional>
#include <string>

#include "io/gimbal/gimbal.hpp"
#include "../model/armor.hpp"
#include "../model/armor_interfaces.hpp"
#include "target.hpp"
#include "tasks/omniperception/detection.hpp"
#include "tools/thread_safe_queue.hpp"

namespace tools
{
class FFTExample;
}

namespace auto_aim
{
class Solver;

class Tracker
{
public:
  /** @brief 使用具体求解器构造目标跟踪器 @param config_path YAML 配置文件路径 @param solver 非拥有求解器指针 */
  Tracker(const std::string & config_path, Solver * solver);
  /** @brief 使用位姿求解接口构造目标跟踪器 @param config_path YAML 配置文件路径 @param solver 非拥有求解器接口指针 */
  Tracker(const std::string & config_path, IArmorPoseSolver * solver);

  /** @brief 获取跟踪状态名称 @return 状态字符串 */
  std::string state() const;

  /** @brief 重置跟踪状态和当前目标 */
  void reset();

  /** @brief 执行哨兵目标跟踪 @param armors 当前帧装甲板列表，函数会原地过滤和求解 @param t 帧时间戳 @param cam_is_short 是否来自短焦相机 @param use_enemy_color 是否过滤敌方颜色 @return 当前有效目标列表 */
  std::list<Target> sb_track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool cam_is_short = true,
    bool use_enemy_color = true);

  /** @brief 执行普通目标跟踪 @param armors 当前帧装甲板列表，函数会原地过滤和求解 @param t 帧时间戳 @param cam_is_short 是否来自短焦相机 @param use_enemy_color 是否过滤敌方颜色 @return 当前有效目标列表 */
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t, 
    bool cam_is_short = true,
    bool use_enemy_color = true);

  /** @brief 执行测试用目标跟踪流程 @param armors 当前帧装甲板列表 @param t 帧时间戳 @param cam_is_short 是否来自短焦相机 @param use_enemy_color 是否过滤敌方颜色 @return 当前有效目标列表 */
  std::list<Target> test_track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t, 
    bool cam_is_short = true,
    bool use_enemy_color = true);


  /** @brief 融合全向感知与自瞄检测结果进行跟踪 @param detection_queue 全向感知结果 @param armors 自瞄装甲板列表 @param t 帧时间戳 @param use_enemy_color 是否过滤敌方颜色 @return 选中的全向检测结果和目标列表 */
  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

  /** @brief 切换到具体求解器 @param solver 非拥有求解器指针 */
  void setSolver(Solver * solver);
  /** @brief 切换到位姿求解接口 @param solver 非拥有求解器接口指针 */
  void setPoseSolver(IArmorPoseSolver * solver);
  /** @brief 绑定云台状态源 @param gimbal 非拥有云台指针 */
  void set_gimbal(io::Gimbal* gimbal) { gimbal_ = gimbal; }
  /** @brief 绑定周期运动分析器 @param fft 非拥有分析器指针 */
  void set_fft(tools::FFTExample * fft);
  /** @brief 获取当前目标累计更新次数 @return 更新次数 */
  inline size_t get_update_count(){return this->target_.update_count_;}
private:
  IArmorPoseSolver * solver_;
  io::Gimbal* gimbal_ = nullptr; // 新增一个云台指针，默认为空
  tools::FFTExample * fft_ = nullptr;  // non-owning
  Color enemy_color_;
  std::string enemy_color_str_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;
  std::optional<uint8_t> last_mode_;
  bool cam_is_switch = false, last_cam_is_short = true;

  /** @brief 根据本帧是否匹配目标推进跟踪状态机 @param found 是否找到匹配装甲板 */
  void state_machine(bool found);

  /** @brief 从装甲板列表初始化新目标 @param armors 候选装甲板列表 @param t 帧时间戳 @return 成功建立目标时返回 true */
  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  /** @brief 使用候选装甲板更新当前目标 @param armors 候选装甲板列表 @param t 帧时间戳 @return 成功匹配时返回 true */
  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  /** @brief 重置周期运动采样状态 */
  void reset_fft_sample_state();
  /** @brief 向周期运动分析器添加装甲板样本 @param armor 匹配装甲板 @param t 帧时间戳 */
  void update_fft_sample(const Armor & armor, std::chrono::steady_clock::time_point t);

  
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
