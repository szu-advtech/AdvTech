# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (FITing_Tree)

set(CMAKE_CXX_STANDARD 14)

# 编译指导
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE -Ofast)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")


# 指定生成目标
add_executable(main_latency main_latency.cc bptree.h db_impl.h shrinkingcone_segmentation.h)
add_executable(main_goodput main_goodput.cc bptree.h db_impl.h shrinkingcone_segmentation.h)
add_executable(main_workload main_workload.cc bptree.h db_impl.h shrinkingcone_segmentation.h)
