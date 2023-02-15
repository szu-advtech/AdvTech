# Install script for directory: /home/code/openvslam-comments/src/openvslam

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/libopenvslam.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/lib" TYPE SHARED_LIBRARY FILES "/home/code/openvslam-comments/build/lib/libopenvslam.so")
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so"
         OLD_RPATH "/usr/local/cuda/lib64:/home/software/libtorch1.10.1cu102/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/lib/libopenvslam.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/openvslam/CUDATSDFIntegrator.h;/usr/local/include/openvslam/PointCloude.h;/usr/local/include/openvslam/Utils.h;/usr/local/include/openvslam/Voxel.h;/usr/local/include/openvslam/config.h;/usr/local/include/openvslam/global_optimization_module.h;/usr/local/include/openvslam/mapping_module.h;/usr/local/include/openvslam/pointcloudmapping.h;/usr/local/include/openvslam/system.h;/usr/local/include/openvslam/tracking_module.h;/usr/local/include/openvslam/type.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam" TYPE FILE FILES
    "/home/code/openvslam-comments/src/openvslam/CUDATSDFIntegrator.h"
    "/home/code/openvslam-comments/src/openvslam/PointCloude.h"
    "/home/code/openvslam-comments/src/openvslam/Utils.h"
    "/home/code/openvslam-comments/src/openvslam/Voxel.h"
    "/home/code/openvslam-comments/src/openvslam/config.h"
    "/home/code/openvslam-comments/src/openvslam/global_optimization_module.h"
    "/home/code/openvslam-comments/src/openvslam/mapping_module.h"
    "/home/code/openvslam-comments/src/openvslam/pointcloudmapping.h"
    "/home/code/openvslam-comments/src/openvslam/system.h"
    "/home/code/openvslam-comments/src/openvslam/tracking_module.h"
    "/home/code/openvslam-comments/src/openvslam/type.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/openvslam/3rd/json/include/nlohmann/json.hpp;/usr/local/include/openvslam/3rd/json/include/nlohmann/json_fwd.hpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam/3rd/json/include/nlohmann" TYPE FILE FILES
    "/home/code/openvslam-comments/3rd/json/include/nlohmann/json.hpp"
    "/home/code/openvslam-comments/3rd/json/include/nlohmann/json_fwd.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/openvslam/3rd/spdlog/include")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam/3rd/spdlog" TYPE DIRECTORY FILES "/home/code/openvslam-comments/3rd/spdlog/include")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/code/openvslam-comments/build/src/openvslam/camera/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/data/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/feature/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/initialize/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/io/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/match/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/module/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/optimize/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/publish/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/solve/cmake_install.cmake")
  include("/home/code/openvslam-comments/build/src/openvslam/util/cmake_install.cmake")

endif()

