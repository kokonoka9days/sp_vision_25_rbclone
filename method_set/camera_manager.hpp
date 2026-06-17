#ifndef IO__CAMERA_MANAGER_HPP
#define IO__CAMERA_MANAGER_HPP

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include "../io/camera.hpp"
#include "../tools/logger.hpp"
#include "../tasks/auto_aim/solver.hpp"
#include "../tasks/auto_aim/planner/planner.hpp"

namespace io {
class CameraManager {
public:
    CameraManager() : daemon_quit_(false) {}
    
    ~CameraManager() {
        
    }

    // 注册受管理的相机
    void add_omn_camera(io::Camera* cam, auto_aim::Solver* solver) {
        if (cam != nullptr && solver != nullptr) {
            omn_cameras_.push_back(std::pair(cam, solver));
        }
    }

    void add_aim_camera(io::Camera* cam, auto_aim::Solver * solver, auto_aim::Planner planner){
        if (cam != nullptr && solver != nullptr) {
            aim_cameras_.push_back(std::tuple(cam, solver, planner));
        }
    }

    // 启动统一守护线程
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