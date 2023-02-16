# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(googletest-download NONE)

include(ExternalProject)
ExternalProject_Add(googletest-download
                    URL;https://github.com/google/googletest/archive/release-1.10.0.tar.gz;URL_HASH;SHA1=9c89be7df9c5e8cb0bc20b3c4b39bf7e82686770;DOWNLOAD_NO_PROGRESS;YES
                    SOURCE_DIR          "/home/code/openvslam-comments/build/googletest-src"
                    BINARY_DIR          "/home/code/openvslam-comments/build/googletest-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)

