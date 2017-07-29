add_definitions( -ggdb3 -std=gnu99 )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(orbslam_writer)

add_executable(se-vins-writer vins_writer.cpp  )
pods_use_pkg_config_packages(se-vins-writer
  image_io_utils opencv eigen3)
target_link_libraries(se-vins-writer boost_filesystem boost_system)
pods_install_executables(se-vins-writer )