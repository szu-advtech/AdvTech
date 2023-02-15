cmake_minimum_required(VERSION 3.17)
project(FuXian)

set(CMAKE_CXX_STANDARD 14)
find_package(PCL CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(tinyply CONFIG REQUIRED)
find_package(embree 3 CONFIG REQUIRED)

add_executable(FuXian main.cpp utils/tools.cpp utils/tools.h entity/point.cpp entity/point.h entity/sample_point.cpp entity/sample_point.h
        entity/sample_point.cpp entity/sample_point.h entity/pos_pack.cpp entity/pos_pack.h entity/view_point.cpp entity/view_point.h
        utils/io_utils.cpp utils/io_utils.h utils/Params.h utils/score.cpp utils/score.h utils/compare.cpp utils/compare.h entity/view_score.cpp entity/view_score.h utils/ea_utils.cpp utils/ea_utils.h utils/ply_utils.cpp utils/ply_utils.h entity/map.cpp entity/map.h utils/log.cpp utils/log.h)
target_link_libraries(FuXian PRIVATE ${PCL_LIBRARIES} CGAL::CGAL tinyply embree)