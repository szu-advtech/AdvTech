# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.6)
   message(FATAL_ERROR "CMake >= 2.6.0 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.6...3.21)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_targetsDefined)
set(_targetsNotDefined)
set(_expectedTargets)
foreach(_expectedTarget openvslam::openvslam)
  list(APPEND _expectedTargets ${_expectedTarget})
  if(NOT TARGET ${_expectedTarget})
    list(APPEND _targetsNotDefined ${_expectedTarget})
  endif()
  if(TARGET ${_expectedTarget})
    list(APPEND _targetsDefined ${_expectedTarget})
  endif()
endforeach()
if("${_targetsDefined}" STREQUAL "${_expectedTargets}")
  unset(_targetsDefined)
  unset(_targetsNotDefined)
  unset(_expectedTargets)
  set(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT "${_targetsDefined}" STREQUAL "")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_targetsDefined}\nTargets not yet defined: ${_targetsNotDefined}\n")
endif()
unset(_targetsDefined)
unset(_targetsNotDefined)
unset(_expectedTargets)


# Compute the installation prefix relative to this file.
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

# Create imported target openvslam::openvslam
add_library(openvslam::openvslam SHARED IMPORTED)

set_target_properties(openvslam::openvslam PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "USE_DBOW2"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/openvslam/3rd/json/include;${_IMPORT_PREFIX}/include/openvslam/3rd/spdlog/include;/include;${_IMPORT_PREFIX}/include/"
  INTERFACE_LINK_LIBRARIES "/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcudadevrt.a;torch;torch_library;/home/software/libtorch1.10.1cu102/lib/libc10.so;/home/software/libtorch1.10.1cu102/lib/libkineto.a;/usr/local/cuda/lib64/stubs/libcuda.so;/usr/local/cuda/lib64/libnvrtc.so;/usr/local/cuda/lib64/libnvToolsExt.so;/usr/local/cuda/lib64/libcudart.so;/home/software/libtorch1.10.1cu102/lib/libc10_cuda.so;/usr/local/cuda/lib64/libcudart.so;Threads::Threads;/usr/lib/x86_64-linux-gnu/libboost_system.so;/usr/lib/x86_64-linux-gnu/libboost_filesystem.so;/usr/lib/x86_64-linux-gnu/libboost_thread.so;-lpthread;/usr/lib/x86_64-linux-gnu/libboost_date_time.so;/usr/lib/x86_64-linux-gnu/libboost_iostreams.so;/usr/lib/x86_64-linux-gnu/libboost_serialization.so;/usr/lib/x86_64-linux-gnu/libboost_chrono.so;/usr/lib/x86_64-linux-gnu/libboost_atomic.so;/usr/lib/x86_64-linux-gnu/libboost_regex.so;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_common.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_common.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_octree.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_octree.so>;/usr/lib/libOpenNI.so;/usr/lib/libOpenNI2.so;vtkChartsCore;vtkCommonColor;vtkCommonDataModel;vtkCommonMath;vtkCommonCore;vtksys;vtkCommonMisc;vtkCommonSystem;vtkCommonTransforms;vtkInfovisCore;vtkFiltersExtraction;vtkCommonExecutionModel;vtkFiltersCore;vtkFiltersGeneral;vtkCommonComputationalGeometry;vtkFiltersStatistics;vtkImagingFourier;vtkImagingCore;vtkalglib;vtkRenderingContext2D;vtkRenderingCore;vtkFiltersGeometry;vtkFiltersSources;vtkRenderingFreeType;/usr/lib/x86_64-linux-gnu/libfreetype.so;/usr/lib/x86_64-linux-gnu/libz.so;vtkftgl;vtkDICOMParser;vtkDomainsChemistry;vtkIOXML;vtkIOGeometry;vtkIOCore;vtkIOXMLParser;/usr/lib/x86_64-linux-gnu/libexpat.so;vtkFiltersAMR;vtkParallelCore;vtkIOLegacy;vtkFiltersFlowPaths;vtkFiltersGeneric;vtkFiltersHybrid;vtkImagingSources;vtkFiltersHyperTree;vtkFiltersImaging;vtkImagingGeneral;vtkFiltersModeling;vtkFiltersParallel;vtkFiltersParallelFlowPaths;vtkParallelMPI;vtkFiltersParallelGeometry;vtkFiltersParallelImaging;vtkFiltersParallelMPI;vtkFiltersParallelStatistics;vtkFiltersProgrammable;vtkFiltersPython;/usr/lib/x86_64-linux-gnu/libpython2.7.so;vtkWrappingPythonCore;vtkWrappingTools;vtkFiltersReebGraph;vtkFiltersSMP;vtkFiltersSelection;vtkFiltersTexture;vtkFiltersVerdict;verdict;vtkGUISupportQt;vtkInteractionStyle;vtkRenderingOpenGL;vtkImagingHybrid;vtkIOImage;vtkmetaio;/usr/lib/x86_64-linux-gnu/libjpeg.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libtiff.so;vtkGUISupportQtOpenGL;vtkGUISupportQtSQL;vtkIOSQL;sqlite3;vtkGUISupportQtWebkit;vtkViewsQt;vtkViewsInfovis;vtkInfovisLayout;vtkInfovisBoostGraphAlgorithms;vtkRenderingLabel;vtkViewsCore;vtkInteractionWidgets;vtkRenderingAnnotation;vtkImagingColor;vtkRenderingVolume;vtkGeovisCore;/usr/lib/x86_64-linux-gnu/libproj.so;vtkIOAMR;/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so;/usr/lib/x86_64-linux-gnu/libsz.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libm.so;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so;vtkIOEnSight;vtkIOExodus;vtkexoIIc;/usr/lib/x86_64-linux-gnu/libnetcdf_c++.so;/usr/lib/x86_64-linux-gnu/libnetcdf.so;vtkIOExport;vtkRenderingGL2PS;vtkRenderingContextOpenGL;/usr/lib/x86_64-linux-gnu/libgl2ps.so;vtkIOFFMPEG;vtkIOMovie;/usr/lib/x86_64-linux-gnu/libtheoraenc.so;/usr/lib/x86_64-linux-gnu/libtheoradec.so;/usr/lib/x86_64-linux-gnu/libogg.so;vtkIOGDAL;vtkIOGeoJSON;vtkIOImport;vtkIOInfovis;/usr/lib/x86_64-linux-gnu/libxml2.so;vtkIOLSDyna;vtkIOMINC;vtkIOMPIImage;vtkIOMPIParallel;vtkIOParallel;vtkIONetCDF;/usr/lib/x86_64-linux-gnu/libjsoncpp.so;vtkIOMySQL;vtkIOODBC;vtkIOPLY;vtkIOParallelExodus;vtkIOParallelLSDyna;vtkIOParallelNetCDF;vtkIOParallelXML;vtkIOPostgreSQL;vtkIOVPIC;VPIC;vtkIOVideo;vtkIOXdmf2;vtkxdmf2;vtkImagingMath;vtkImagingMorphological;vtkImagingStatistics;vtkImagingStencil;vtkInteractionImage;vtkLocalExample;vtkParallelMPI4Py;vtkPythonInterpreter;vtkRenderingExternal;vtkRenderingFreeTypeFontConfig;vtkRenderingImage;vtkRenderingLIC;vtkRenderingLOD;vtkRenderingMatplotlib;vtkRenderingParallel;vtkRenderingParallelLIC;vtkRenderingQt;vtkRenderingVolumeAMR;vtkRenderingVolumeOpenGL;vtkTestingGenericBridge;vtkTestingIOSQL;vtkTestingRendering;vtkViewsContext2D;vtkViewsGeovis;vtkWrappingJava;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_io.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_io.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libflann_cpp_s.a>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libflann_cpp_s.a>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_kdtree.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_kdtree.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_search.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_search.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_filters.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_filters.so>;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libpcl_visualization.so>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libpcl_visualization.so>;/usr/lib/x86_64-linux-gnu/libboost_system.so;/usr/lib/x86_64-linux-gnu/libboost_filesystem.so;/usr/lib/x86_64-linux-gnu/libboost_thread.so;-lpthread;/usr/lib/x86_64-linux-gnu/libboost_date_time.so;/usr/lib/x86_64-linux-gnu/libboost_iostreams.so;/usr/lib/x86_64-linux-gnu/libboost_serialization.so;/usr/lib/x86_64-linux-gnu/libboost_chrono.so;/usr/lib/x86_64-linux-gnu/libboost_atomic.so;/usr/lib/x86_64-linux-gnu/libboost_regex.so;/usr/lib/libOpenNI.so;/usr/lib/libOpenNI2.so;\$<\$<NOT:\$<CONFIG:DEBUG>>:/usr/lib/x86_64-linux-gnu/libflann_cpp_s.a>;\$<\$<CONFIG:DEBUG>:/usr/lib/x86_64-linux-gnu/libflann_cpp_s.a>;vtkChartsCore;vtkCommonColor;vtkCommonDataModel;vtkCommonMath;vtkCommonCore;vtksys;vtkCommonMisc;vtkCommonSystem;vtkCommonTransforms;vtkInfovisCore;vtkFiltersExtraction;vtkCommonExecutionModel;vtkFiltersCore;vtkFiltersGeneral;vtkCommonComputationalGeometry;vtkFiltersStatistics;vtkImagingFourier;vtkImagingCore;vtkalglib;vtkRenderingContext2D;vtkRenderingCore;vtkFiltersGeometry;vtkFiltersSources;vtkRenderingFreeType;/usr/lib/x86_64-linux-gnu/libfreetype.so;/usr/lib/x86_64-linux-gnu/libz.so;vtkftgl;vtkDICOMParser;vtkDomainsChemistry;vtkIOXML;vtkIOGeometry;vtkIOCore;vtkIOXMLParser;/usr/lib/x86_64-linux-gnu/libexpat.so;vtkFiltersAMR;vtkParallelCore;vtkIOLegacy;vtkFiltersFlowPaths;vtkFiltersGeneric;vtkFiltersHybrid;vtkImagingSources;vtkFiltersHyperTree;vtkFiltersImaging;vtkImagingGeneral;vtkFiltersModeling;vtkFiltersParallel;vtkFiltersParallelFlowPaths;vtkParallelMPI;vtkFiltersParallelGeometry;vtkFiltersParallelImaging;vtkFiltersParallelMPI;vtkFiltersParallelStatistics;vtkFiltersProgrammable;vtkFiltersPython;/usr/lib/x86_64-linux-gnu/libpython2.7.so;vtkWrappingPythonCore;vtkWrappingTools;vtkFiltersReebGraph;vtkFiltersSMP;vtkFiltersSelection;vtkFiltersTexture;vtkFiltersVerdict;verdict;vtkGUISupportQt;vtkInteractionStyle;vtkRenderingOpenGL;vtkImagingHybrid;vtkIOImage;vtkmetaio;/usr/lib/x86_64-linux-gnu/libjpeg.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libtiff.so;vtkGUISupportQtOpenGL;vtkGUISupportQtSQL;vtkIOSQL;sqlite3;vtkGUISupportQtWebkit;vtkViewsQt;vtkViewsInfovis;vtkInfovisLayout;vtkInfovisBoostGraphAlgorithms;vtkRenderingLabel;vtkViewsCore;vtkInteractionWidgets;vtkRenderingAnnotation;vtkImagingColor;vtkRenderingVolume;vtkGeovisCore;/usr/lib/x86_64-linux-gnu/libproj.so;vtkIOAMR;/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so;/usr/lib/x86_64-linux-gnu/libsz.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libm.so;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so;vtkIOEnSight;vtkIOExodus;vtkexoIIc;/usr/lib/x86_64-linux-gnu/libnetcdf_c++.so;/usr/lib/x86_64-linux-gnu/libnetcdf.so;vtkIOExport;vtkRenderingGL2PS;vtkRenderingContextOpenGL;/usr/lib/x86_64-linux-gnu/libgl2ps.so;vtkIOFFMPEG;vtkIOMovie;/usr/lib/x86_64-linux-gnu/libtheoraenc.so;/usr/lib/x86_64-linux-gnu/libtheoradec.so;/usr/lib/x86_64-linux-gnu/libogg.so;vtkIOGDAL;vtkIOGeoJSON;vtkIOImport;vtkIOInfovis;/usr/lib/x86_64-linux-gnu/libxml2.so;vtkIOLSDyna;vtkIOMINC;vtkIOMPIImage;vtkIOMPIParallel;vtkIOParallel;vtkIONetCDF;/usr/lib/x86_64-linux-gnu/libjsoncpp.so;vtkIOMySQL;vtkIOODBC;vtkIOPLY;vtkIOParallelExodus;vtkIOParallelLSDyna;vtkIOParallelNetCDF;vtkIOParallelXML;vtkIOPostgreSQL;vtkIOVPIC;VPIC;vtkIOVideo;vtkIOXdmf2;vtkxdmf2;vtkImagingMath;vtkImagingMorphological;vtkImagingStatistics;vtkImagingStencil;vtkInteractionImage;vtkLocalExample;vtkParallelMPI4Py;vtkPythonInterpreter;vtkRenderingExternal;vtkRenderingFreeTypeFontConfig;vtkRenderingImage;vtkRenderingLIC;vtkRenderingLOD;vtkRenderingMatplotlib;vtkRenderingParallel;vtkRenderingParallelLIC;vtkRenderingQt;vtkRenderingVolumeAMR;vtkRenderingVolumeOpenGL;vtkTestingGenericBridge;vtkTestingIOSQL;vtkTestingRendering;vtkViewsContext2D;vtkViewsGeovis;vtkWrappingJava;Eigen3::Eigen;yaml-cpp;opencv_core;opencv_features2d;opencv_calib3d;g2o::core;g2o::stuff;g2o::types_sba;g2o::types_sim3;g2o::solver_dense;g2o::solver_eigen;g2o::solver_csparse;g2o::csparse_extension;/usr/lib/x86_64-linux-gnu/libccolamd.so;/usr/lib/x86_64-linux-gnu/libcamd.so;/usr/lib/x86_64-linux-gnu/libcolamd.so;/usr/lib/x86_64-linux-gnu/libamd.so;/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libblas.so;/usr/lib/x86_64-linux-gnu/libf77blas.so;/usr/lib/x86_64-linux-gnu/libatlas.so;/usr/lib/x86_64-linux-gnu/libblas.so;/usr/lib/x86_64-linux-gnu/libf77blas.so;/usr/lib/x86_64-linux-gnu/libatlas.so;/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so;/usr/lib/x86_64-linux-gnu/librt.so;/lib/libdbow2.so;/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libblas.so;/usr/lib/x86_64-linux-gnu/libf77blas.so;/usr/lib/x86_64-linux-gnu/libatlas.so"
)

if(CMAKE_VERSION VERSION_LESS 2.8.12)
  message(FATAL_ERROR "This file relies on consumers using CMake 2.8.12 or greater.")
endif()

# Load information for each installed configuration.
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/openvslamTargets-*.cmake")
foreach(f ${CONFIG_FILES})
  include(${f})
endforeach()

# Cleanup temporary variables.
set(_IMPORT_PREFIX)

# Loop over all imported files and verify that they actually exist
foreach(target ${_IMPORT_CHECK_TARGETS} )
  foreach(file ${_IMPORT_CHECK_FILES_FOR_${target}} )
    if(NOT EXISTS "${file}" )
      message(FATAL_ERROR "The imported target \"${target}\" references the file
   \"${file}\"
but this file does not exist.  Possible reasons include:
* The file was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and contained
   \"${CMAKE_CURRENT_LIST_FILE}\"
but not all the files it references.
")
    endif()
  endforeach()
  unset(_IMPORT_CHECK_FILES_FOR_${target})
endforeach()
unset(_IMPORT_CHECK_TARGETS)

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
