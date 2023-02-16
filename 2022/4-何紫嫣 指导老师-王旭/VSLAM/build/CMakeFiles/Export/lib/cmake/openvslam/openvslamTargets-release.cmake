#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvslam::openvslam" for configuration "Release"
set_property(TARGET openvslam::openvslam APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(openvslam::openvslam PROPERTIES
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libopenvslam.so"
  IMPORTED_SONAME_RELEASE "libopenvslam.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS openvslam::openvslam )
list(APPEND _IMPORT_CHECK_FILES_FOR_openvslam::openvslam "/usr/local/lib/libopenvslam.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
