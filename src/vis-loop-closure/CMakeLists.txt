set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################
set ( OpenCV_LIBS opencv_core
                  opencv_highgui
                  opencv_nonfree
                opencv_calib3d
                opencv_imgproc
                          opencv_features2d )

# Was called drc-registeration (sic)
add_library(vis-loop-closure  SHARED vis-loop-closure.cpp)
pods_use_pkg_config_packages(vis-loop-closure
  opencv
  lcmtypes_bot2-core bot2-lcmgl-client
  pronto_vis bot2-frames   bot2-frames_cpp voconfig vofeatures
  image_io_utils
  eigen3 estimate-pose)
target_link_libraries (vis-loop-closure ${OpenCV_LIBS})
set_target_properties(vis-loop-closure PROPERTIES SOVERSION 1)
pods_install_libraries(vis-loop-closure)
pods_install_headers( vis-loop-closure.hpp DESTINATION drcvision)
pods_install_pkg_config_file(vis-loop-closure
  LIBS -lvis-loop-closure 
  VERSION 0.0.1)


################
# Was called drc-registeration-batch (sic)
add_executable(drc-vis-loop-closure-batch vis-loop-closure-batch.cpp)
target_link_libraries(drc-vis-loop-closure-batch boost_system)
pods_use_pkg_config_packages(drc-vis-loop-closure-batch
  bot2-vis 
  path-util
  pronto_vis opencv
  lcm bot2-lcmgl-client vis-loop-closure  )
pods_install_executables(drc-vis-loop-closure-batch)
