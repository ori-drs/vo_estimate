add_definitions( -ggdb3 -std=c++11  )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tools)

add_executable(se-vo-slam vo_slam.cpp)
pods_use_pkg_config_packages(se-vo-slam path-util   eigen3  pronto_vis)
target_link_libraries(se-vo-slam lcm boost_system boost_thread )
pods_install_executables(se-vo-slam)


###############
add_executable(se-simple-isam-demo simple_isam_demo.cpp)
pods_use_pkg_config_packages(se-simple-isam-demo
    pronto_vis pronto_math  eigen3 )
target_link_libraries (se-simple-isam-demo isam cholmod  )
pods_install_executables(se-simple-isam-demo)

###############
add_executable(se-isam-lidar isam_read_file_lidar.cpp)
pods_use_pkg_config_packages(se-isam-lidar
    pronto_vis pronto_math  eigen3 )
target_link_libraries (se-isam-lidar isam cholmod  )
pods_install_executables(se-isam-lidar)
