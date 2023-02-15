# Install script for directory: /home/code/openvslam-comments/src/openvslam/module

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
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/openvslam/module/frame_tracker.h;/usr/local/include/openvslam/module/initializer.h;/usr/local/include/openvslam/module/keyframe_inserter.h;/usr/local/include/openvslam/module/local_map_cleaner.h;/usr/local/include/openvslam/module/local_map_updater.h;/usr/local/include/openvslam/module/loop_bundle_adjuster.h;/usr/local/include/openvslam/module/loop_detector.h;/usr/local/include/openvslam/module/relocalizer.h;/usr/local/include/openvslam/module/two_view_triangulator.h;/usr/local/include/openvslam/module/type.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam/module" TYPE FILE FILES
    "/home/code/openvslam-comments/src/openvslam/module/frame_tracker.h"
    "/home/code/openvslam-comments/src/openvslam/module/initializer.h"
    "/home/code/openvslam-comments/src/openvslam/module/keyframe_inserter.h"
    "/home/code/openvslam-comments/src/openvslam/module/local_map_cleaner.h"
    "/home/code/openvslam-comments/src/openvslam/module/local_map_updater.h"
    "/home/code/openvslam-comments/src/openvslam/module/loop_bundle_adjuster.h"
    "/home/code/openvslam-comments/src/openvslam/module/loop_detector.h"
    "/home/code/openvslam-comments/src/openvslam/module/relocalizer.h"
    "/home/code/openvslam-comments/src/openvslam/module/two_view_triangulator.h"
    "/home/code/openvslam-comments/src/openvslam/module/type.h"
    )
endif()

