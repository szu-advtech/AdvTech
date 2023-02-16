cmake_minimum_required(VERSION 3.0.0)
project(HyGCN)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(./dramsim3)

file(GLOB_RECURSE hygcn_src "hygcn/*.cc" "hygcn/*.cpp" "hygcn/*.c" "hygcn/*.h")
add_library(HyGCN STATIC ${hygcn_src})
target_link_libraries(HyGCN PRIVATE args dramsim3)
target_include_directories(HyGCN PRIVATE hygcn dramsim3/src dramsim3/ext)

add_executable(hygcntest hygcn_test/test.cpp)
target_include_directories(hygcntest PRIVATE hygcn dramsim3/src dramsim3/ext/)
target_link_libraries(hygcntest PRIVATE dramsim3 HyGCN args)
target_compile_options(hygcntest PRIVATE)
set_target_properties(hygcntest PROPERTIES
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )
