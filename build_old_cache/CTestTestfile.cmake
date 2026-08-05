# CMake generated Testfile for 
# Source directory: /home/rm/Desktop/sp_vision_25_rbclone
# Build directory: /home/rm/Desktop/sp_vision_25_rbclone/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(laser_ray_calibration_test "/home/rm/Desktop/sp_vision_25_rbclone/build/laser_ray_calibration_test")
set_tests_properties(laser_ray_calibration_test PROPERTIES  _BACKTRACE_TRIPLES "/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;145;add_test;/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;0;")
add_test(drone_planner_laser_test "/home/rm/Desktop/sp_vision_25_rbclone/build/drone_planner_laser_test")
set_tests_properties(drone_planner_laser_test PROPERTIES  _BACKTRACE_TRIPLES "/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;146;add_test;/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;0;")
add_test(drone_planner_inline_config_test "/home/rm/Desktop/sp_vision_25_rbclone/build/drone_planner_inline_config_test")
set_tests_properties(drone_planner_inline_config_test PROPERTIES  _BACKTRACE_TRIPLES "/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;147;add_test;/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;0;")
add_test(prediction_cadence_test "/home/rm/Desktop/sp_vision_25_rbclone/build/prediction_cadence_test")
set_tests_properties(prediction_cadence_test PROPERTIES  _BACKTRACE_TRIPLES "/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;148;add_test;/home/rm/Desktop/sp_vision_25_rbclone/CMakeLists.txt;0;")
subdirs("tools")
subdirs("io")
subdirs("tasks/auto_drone")
