#!/bin/sh
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE="RelWithDebInfo" ..
cmake --build . -j --clean-first
