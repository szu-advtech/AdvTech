# CMake generated Testfile for 
# Source directory: /home/code/openvslam-comments/test
# Build directory: /home/code/openvslam-comments/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_openvslam_data_common_get_cell_indices "/home/code/openvslam-comments/build/test/test_openvslam_data_common_get_cell_indices")
set_tests_properties(test_openvslam_data_common_get_cell_indices PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_feature_orb_extractor "/home/code/openvslam-comments/build/test/test_openvslam_feature_orb_extractor")
set_tests_properties(test_openvslam_feature_orb_extractor PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_feature_orb_params "/home/code/openvslam-comments/build/test/test_openvslam_feature_orb_params")
set_tests_properties(test_openvslam_feature_orb_params PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_match_angle_checker "/home/code/openvslam-comments/build/test/test_openvslam_match_angle_checker")
set_tests_properties(test_openvslam_match_angle_checker PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_match_base "/home/code/openvslam-comments/build/test/test_openvslam_match_base")
set_tests_properties(test_openvslam_match_base PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_solve_essential_solver "/home/code/openvslam-comments/build/test/test_openvslam_solve_essential_solver")
set_tests_properties(test_openvslam_solve_essential_solver PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_solve_fundamental_solver "/home/code/openvslam-comments/build/test/test_openvslam_solve_fundamental_solver")
set_tests_properties(test_openvslam_solve_fundamental_solver PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_solve_homography_solver "/home/code/openvslam-comments/build/test/test_openvslam_solve_homography_solver")
set_tests_properties(test_openvslam_solve_homography_solver PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_solve_pnp_solver "/home/code/openvslam-comments/build/test/test_openvslam_solve_pnp_solver")
set_tests_properties(test_openvslam_solve_pnp_solver PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_util_fancy_index "/home/code/openvslam-comments/build/test/test_openvslam_util_fancy_index")
set_tests_properties(test_openvslam_util_fancy_index PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_util_random_array "/home/code/openvslam-comments/build/test/test_openvslam_util_random_array")
set_tests_properties(test_openvslam_util_random_array PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
add_test(test_openvslam_util_trigonometric "/home/code/openvslam-comments/build/test/test_openvslam_util_trigonometric")
set_tests_properties(test_openvslam_util_trigonometric PROPERTIES  _BACKTRACE_TRIPLES "/home/code/openvslam-comments/test/CMakeLists.txt;50;add_test;/home/code/openvslam-comments/test/CMakeLists.txt;0;")
subdirs("../googletest-build")
subdirs("helper")
