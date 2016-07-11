add_definitions( -ggdb3 -std=gnu99 )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tools)

add_executable(se-vo-slam vo_slam.cpp)
pods_use_pkg_config_packages(se-vo-slam path-util   eigen3  pronto_vis)
target_link_libraries(se-vo-slam lcm boost_system boost_thread )
pods_install_executables(se-vo-slam)