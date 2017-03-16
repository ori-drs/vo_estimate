add_definitions( -ggdb3 -std=c++11  )

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(tools)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(PCL 1.7 REQUIRED)#COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(PCL_LIBRARIES pcl_io-1.7 pcl_common-1.7 pcl_filters-1.7 pcl_features-1.7 pcl_io-1.7 pcl_surface-1.7 pcl_segmentation-1.7 pcl_visualization-1.7 pcl_search-1.7 pcl_kdtree-1.7 pcl_keypoints-1.7)

add_executable(se-vo-slam vo_slam.cpp)
pods_use_pkg_config_packages(se-vo-slam path-util   eigen3  pronto_vis)
target_link_libraries(se-vo-slam lcm boost_system boost_thread )
pods_install_executables(se-vo-slam)

############## vo slam w/ relative lidar points
add_executable(se-vo-slam-lidar vo_slam_relative_lidar.cpp)
pods_use_pkg_config_packages(se-vo-slam-lidar path-util   eigen3  pronto_vis bot2-frames ${PCL_LIBRARIES})
target_link_libraries(se-vo-slam-lidar lcm boost_system boost_thread cloud_accumulate pthread  pronto_vis  boost_filesystem ${VTK_LIBRARIES} diftFilters)
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
    pronto_vis pronto_math  eigen3 )
target_link_libraries (se-isam-lidar isam cholmod boost_system )
pods_install_executables(se-isam-lidar)
