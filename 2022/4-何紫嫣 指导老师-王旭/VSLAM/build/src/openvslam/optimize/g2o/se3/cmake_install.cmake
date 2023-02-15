# Install script for directory: /home/code/openvslam-comments/src/openvslam/optimize/g2o/se3

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
   "/usr/local/include/openvslam/optimize/g2o/se3/equirectangular_pose_opt_edge.h;/usr/local/include/openvslam/optimize/g2o/se3/equirectangular_reproj_edge.h;/usr/local/include/openvslam/optimize/g2o/se3/perspective_pose_opt_edge.h;/usr/local/include/openvslam/optimize/g2o/se3/perspective_reproj_edge.h;/usr/local/include/openvslam/optimize/g2o/se3/pose_opt_edge_wrapper.h;/usr/local/include/openvslam/optimize/g2o/se3/reproj_edge_wrapper.h;/usr/local/include/openvslam/optimize/g2o/se3/shot_vertex.h;/usr/local/include/openvslam/optimize/g2o/se3/shot_vertex_container.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam/optimize/g2o/se3" TYPE FILE FILES
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/equirectangular_pose_opt_edge.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/equirectangular_reproj_edge.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/perspective_pose_opt_edge.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/perspective_reproj_edge.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/pose_opt_edge_wrapper.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/reproj_edge_wrapper.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/shot_vertex.h"
    "/home/code/openvslam-comments/src/openvslam/optimize/g2o/se3/shot_vertex_container.h"
    )
endif()

