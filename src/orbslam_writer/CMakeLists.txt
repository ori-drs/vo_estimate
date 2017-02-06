add_definitions( -ggdb3 -std=gnu99 )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(orbslam_writer)


add_executable(se-orbslam-writer orbslam_writer.cpp  )
pods_use_pkg_config_packages(se-orbslam-writer
  image_io_utils opencv eigen3)
pods_install_executables(se-orbslam-writer )