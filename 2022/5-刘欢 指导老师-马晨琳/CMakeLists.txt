cmake_minimum_required(VERSION 3.23)
project(homework)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(homework
        hermit.cc
        trs_tree.cpp
        trs_tree.h bplustree.cpp)

