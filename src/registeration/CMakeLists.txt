set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################
set ( OpenCV_LIBS opencv_core
                  opencv_highgui
                  opencv_nonfree
                opencv_calib3d
                opencv_imgproc
                          opencv_features2d )

add_library(registeration  SHARED registeration.cpp)
pods_use_pkg_config_packages(registeration
  opencv
  lcmtypes_bot2-core bot2-lcmgl-client
  pronto_vis bot2-frames   bot2-frames_cpp voconfig vofeatures
  image_io_utils
  eigen3 estimate-pose)
target_link_libraries (registeration ${OpenCV_LIBS})
set_target_properties(registeration PROPERTIES SOVERSION 1)
pods_install_libraries(registeration)
pods_install_headers( registeration.hpp DESTINATION registeration)
pods_install_pkg_config_file(registeration
  LIBS -lregisteration 
  VERSION 0.0.1)


################
add_executable(drc-registeration-batch registeration-batch.cpp)
target_link_libraries(drc-registeration-batch boost_system)
pods_use_pkg_config_packages(drc-registeration-batch
  bot2-vis 
  path-util
  pronto_vis opencv
  lcm bot2-lcmgl-client registeration  )
pods_install_executables(drc-registeration-batch)
