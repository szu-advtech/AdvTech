cmake_minimum_required(VERSION 3.0.0)
project(FPIM)
set(CMAKE_CXX_STANDARD 17)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_library(inih INTERFACE)
target_include_directories(inih INTERFACE ext/headers)

add_library(format INTERFACE)
target_include_directories(format INTERFACE ext/fmt/include)
target_compile_definitions(format INTERFACE FMT_HEADER_ONLY=1)

add_library(json INTERFACE)
target_include_directories(json INTERFACE ext/headers)

add_library(TOOL STATIC ext/tool/tool.cpp ext/tool/graph.cpp)
target_include_directories(TOOL INTERFACE ext/tool)

# Main DRAMSim Lib
add_library(dramsim3 STATIC
    src/bankstate.cc
    src/channel_state.cc
    src/command_queue.cc
    src/common.cc
    src/configuration.cc
    src/controller.cc
    src/dram_system.cc
    src/hmc.cc
    src/refresh.cc
    src/simple_stats.cc
    src/timing.cc
    src/memory_system.cc

    src/event.h
    src/fpim_stats.h
    src/event.cpp
    src/app/app_config.h
    src/app/app_config.cpp
    src/control/bank_layout.cpp
    src/control/bank_layout.h
    src/control/general_config.h
    src/control/fpim_config.h

    src/hardware/cam.h
    src/hardware/mac.h
    src/hardware/buffer.h
    src/hardware/spm.h
    src/hardware/broadcast_arbiter.h
    src/hardware/fifo.h

    src/logic/group_logic.cpp
    src/logic/group_logic.h
    src/logic/base_logic.h
    src/logic/base_logic.cpp

    src/sched/community.cpp
    src/sched/community.h
    src/app/mapper.h
    src/app/mapper.cpp
    src/app/record.h)

if (THERMAL)
    # dependency check
    # sudo apt-get install libatlas-base-dev on ubuntu
    find_package(BLAS REQUIRED)
    find_package(OpenMP REQUIRED)
    # YOU need to build superlu on your own. Do the following:
    # git submodule update --init
    # cd ext/SuperLU_MT_3.1 && make lib
    find_library(SUPERLU
        NAME superlu_mt_OPENMP libsuperlu_mt_OPENMP
        HINTS ${PROJECT_SOURCE_DIR}/ext/SuperLU_MT_3.1/lib/
    )

    target_link_libraries(dramsim3
        PRIVATE ${SUPERLU} f77blas atlas m ${OpenMP_C_FLAGS}
    )
    target_sources(dramsim3
        PRIVATE src/thermal.cc src/sp_ienv.c src/thermal_solver.c
    )
    target_compile_options(dramsim3 PRIVATE -DTHERMAL -D_LONGINT -DAdd_ ${OpenMP_C_FLAGS})

    add_executable(thermalreplay src/thermal_replay.cc)
    target_link_libraries(thermalreplay dramsim3 inih)
    target_compile_options(thermalreplay PRIVATE -DTHERMAL -D_LONGINT -DAdd_ ${OpenMP_C_FLAGS})
endif (THERMAL)

if (CMD_TRACE)
    target_compile_options(dramsim3 PRIVATE -DCMD_TRACE)
endif (CMD_TRACE)

if (ADDR_TRACE)
    target_compile_options(dramsim3 PRIVATE -DADDR_TRACE)
endif (ADDR_TRACE)


target_include_directories(dramsim3 PRIVATE src ext)
target_compile_options(dramsim3 PRIVATE -Wall)
target_link_libraries(dramsim3 PRIVATE inih format TOOL)
set_target_properties(dramsim3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)


# argparsing library, only used in main program not the library
add_library(args INTERFACE)
target_include_directories(args INTERFACE ext/headers)

# Unit testing
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ext/headers)



# trace CPU, .etc
add_executable(dramsim3main src/main.cc src/cpu.cc)
target_include_directories(dramsim3main PRIVATE ext)
target_link_libraries(dramsim3main PRIVATE dramsim3 args)
target_compile_options(dramsim3main PRIVATE)
set_target_properties(dramsim3main PROPERTIES
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_executable(fpimtest fpimtest/test.cpp)
target_include_directories(fpimtest PRIVATE ext src)
target_link_libraries(fpimtest PRIVATE dramsim3 args)
target_compile_options(fpimtest PRIVATE)
set_target_properties(fpimtest PROPERTIES
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

add_executable(traversal fpimtest/traversal.cpp)
target_include_directories(traversal PRIVATE ext src)
target_link_libraries(traversal PRIVATE dramsim3 args)
target_compile_options(traversal PRIVATE)
set_target_properties(traversal PROPERTIES
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

add_executable(fifo_test fpimtest/fifo_test.cpp)
target_include_directories(fifo_test PRIVATE ext src)
target_link_libraries(fifo_test PRIVATE dramsim3 args)
target_compile_options(fifo_test PRIVATE)
set_target_properties(fifo_test PROPERTIES
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )
add_executable(dfs fpimtest/dfs.cpp)
target_include_directories(dfs PRIVATE ext src)
target_link_libraries(dfs PRIVATE dramsim3 args)
target_compile_options(dfs PRIVATE)
set_target_properties(dfs PROPERTIES
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

#add_executable(dramsim3test EXCLUDE_FROM_ALL
#    tests/test_config.cc
#    tests/test_dramsys.cc
#    tests/test_hmcsys.cc # IDK somehow this can literally crush your computer
#)
#target_link_libraries(dramsim3test Catch dramsim3)
#target_include_directories(dramsim3test PRIVATE src/)
#
## We have to use this custome command because there's a bug in cmake
## that if you do `make test` it doesn't build your updated test files
## so we're stucking with `make dramsim3test` for now
#add_custom_command(
#    TARGET dramsim3test POST_BUILD
#    COMMAND dramsim3test
#    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#    DEPENDS dramsim3test dramsim3
#)
#