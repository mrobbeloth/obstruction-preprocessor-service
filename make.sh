#!/bin/sh
cmake -Bbuild -DCMAKE_BUILD_TYPE=Released
cd build
make
cd ..
