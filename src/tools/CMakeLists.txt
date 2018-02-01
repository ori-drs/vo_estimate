add_definitions( -ggdb3 -std=c++11  )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tools)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(PkgConfig REQUIRED)

pkg_check_modules(PCL_IO pcl_io-1.7)
if(PCL_IO_FOUND)
  find_package(PCL 1.7 REQUIRED)#COMPONENTS common io)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  set(PCL_LIBRARIES pcl_io-1.7 pcl_common-1.7 pcl_filters-1.7 pcl_features-1.7 pcl_io-1.7 pcl_surface-1.7 pcl_segmentation-1.7 pcl_visualization-1.7 pcl_search-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7)
endif()

pkg_check_modules(PCL_IO pcl_io-1.8)
if(PCL_IO_FOUND)
  find_package(PCL 1.8 REQUIRED)#COMPONENTS common io)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  set(PCL_LIBRARIES pcl_io-1.8 pcl_common-1.8 pcl_filters-1.8 pcl_features-1.8 pcl_io-1.8 pcl_surface-1.8 pcl_segmentation-1.8 pcl_visualization-1.8 pcl_search-1.8 pcl_kdtree-1.8 pcl_keypoints-1.8)
endif()

############## vo slam w/ relative lidar points
add_executable(se-vo-slam-lidar vo_slam_relative_lidar.cpp)
pods_use_pkg_config_packages(se-vo-slam-lidar path-util   eigen3  pronto_vis bot2-frames ${PCL_LIBRARIES})
target_link_libraries(se-vo-slam-lidar lcm boost_system boost_thread cloud_accumulate pthread  pronto_vis  boost_filesystem ${VTK_LIBRARIES} pronto_lidar_filters)
pods_install_executables(se-vo-slam-lidar)


###############
add_executable(se-simple-isam-demo simple_isam_demo.cpp)
pods_use_pkg_config_packages(se-simple-isam-demo
    pronto_vis pronto_math  eigen3 )
target_link_libraries (se-simple-isam-demo isam cholmod  )
pods_install_executables(se-simple-isam-demo)

###############
add_executable(se-isam-lidar isam_read_file_lidar.cpp)
pods_use_pkg_config_packages(se-isam-lidar
    pronto_vis pronto_math  eigen3 ${PCL_LIBRARIES})
target_link_libraries (se-isam-lidar isam cholmod boost_system )
pods_install_executables(se-isam-lidar)

###############
add_executable(se-transform-printer transform-printer.cpp)
pods_use_pkg_config_packages(se-transform-printer bot2-frames_cpp
    pronto_vis pronto_math  eigen3 bot2-frames  path-util)
target_link_libraries (se-transform-printer isam cholmod boost_system )
pods_install_executables(se-transform-printer)
