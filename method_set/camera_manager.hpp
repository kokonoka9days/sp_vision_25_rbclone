#ifndef IO__CAMERA_MANAGER_HPP
#define IO__CAMERA_MANAGER_HPP

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include "../io/camera/camera.hpp"
#include "../tools/logger.hpp"
#include "../tasks/auto_aim/geometry/solver.hpp"
#include "../tasks/auto_aim/aiming/planner/planner.hpp"

namespace io {
class CameraManager {
public:
    /** @brief 构造空的相机管理器 */
    CameraManager() : daemon_quit_(false) {}
    
    /** @brief 销毁相机管理器 */
    ~CameraManager() {
        
    }

    /** @brief 注册全向感知相机 @param cam 非拥有相机指针 @param solver 非拥有求解器指针 */
    void add_omn_camera(io::Camera* cam, auto_aim::Solver* solver) {
        if (cam != nullptr && solver != nullptr) {
            omn_cameras_.push_back(std::pair(cam, solver));
        }
    }

    /** @brief 注册自瞄相机 @param cam 非拥有相机指针 @param solver 非拥有求解器指针 @param planner 对应规划器 */
    void add_aim_camera(io::Camera* cam, auto_aim::Solver * solver, auto_aim::Planner planner){
        if (cam != nullptr && solver != nullptr) {
            aim_cameras_.push_back(std::tuple(cam, solver, planner));
        }
    }

    /** @brief 启动统一相机状态守护线程 */
    void start_daemon() {
        daemon_thread_ = std::thread([this]() {
            tools::logger()->info("[CameraManager] Unified camera daemon thread started.");
            while (!daemon_quit_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                for (const auto omn_cam : omn_cameras_) {
                    bool capturing = omn_cam.first->get_capturing();
                    // if(!capturing) omn_cam.first.
                }
            }
        });
    }

private:
    std::vector<std::pair<io::Camera*, auto_aim::Solver *>> omn_cameras_;
    std::vector<std::tuple<io::Camera*, auto_aim::Solver *, auto_aim::Planner>> aim_cameras_;
    std::thread daemon_thread_;
    std::atomic<bool> daemon_quit_;
};
} // namespace io

#endif // IO__CAMERA_MANAGER_HPP
