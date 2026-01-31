#!/bin/bash
cmake -B build
make -C build/ -j`nproc`
# make -C build/ -j4

# ros2 build
# cd sp_ws
# colcon build --packages-select sp_msgs
# source sp_ws/install/setup.bash
# 