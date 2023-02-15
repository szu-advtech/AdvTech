# Install script for directory: /home/code/openvslam-comments/src/openvslam/data

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
   "/usr/local/include/openvslam/data/bow_database.h;/usr/local/include/openvslam/data/bow_vocabulary.h;/usr/local/include/openvslam/data/camera_database.h;/usr/local/include/openvslam/data/common.h;/usr/local/include/openvslam/data/frame.h;/usr/local/include/openvslam/data/frame_statistics.h;/usr/local/include/openvslam/data/graph_node.h;/usr/local/include/openvslam/data/keyframe.h;/usr/local/include/openvslam/data/landmark.h;/usr/local/include/openvslam/data/map_database.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/openvslam/data" TYPE FILE FILES
    "/home/code/openvslam-comments/src/openvslam/data/bow_database.h"
    "/home/code/openvslam-comments/src/openvslam/data/bow_vocabulary.h"
    "/home/code/openvslam-comments/src/openvslam/data/camera_database.h"
    "/home/code/openvslam-comments/src/openvslam/data/common.h"
    "/home/code/openvslam-comments/src/openvslam/data/frame.h"
    "/home/code/openvslam-comments/src/openvslam/data/frame_statistics.h"
    "/home/code/openvslam-comments/src/openvslam/data/graph_node.h"
    "/home/code/openvslam-comments/src/openvslam/data/keyframe.h"
    "/home/code/openvslam-comments/src/openvslam/data/landmark.h"
    "/home/code/openvslam-comments/src/openvslam/data/map_database.h"
    )
endif()

