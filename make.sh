#!/bin/bash
cmake -B build
# make -C build/ -j`nproc`
make -C build/ -j4