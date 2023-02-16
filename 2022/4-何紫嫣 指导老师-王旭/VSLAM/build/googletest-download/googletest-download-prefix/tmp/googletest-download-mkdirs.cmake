# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/code/openvslam-comments/build/googletest-src"
  "/home/code/openvslam-comments/build/googletest-build"
  "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix"
  "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix/tmp"
  "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix/src/googletest-download-stamp"
  "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix/src"
  "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix/src/googletest-download-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/code/openvslam-comments/build/googletest-download/googletest-download-prefix/src/googletest-download-stamp/${subDir}")
endforeach()
