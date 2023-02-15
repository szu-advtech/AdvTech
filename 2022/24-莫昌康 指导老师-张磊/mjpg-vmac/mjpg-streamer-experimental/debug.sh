#!/bin/bash

make distclean
make CMAKE_BUILD_TYPE=Debug
sudo make install
