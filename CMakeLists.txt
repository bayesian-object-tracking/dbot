cmake_minimum_required(VERSION 2.8.3)
project(state_filtering)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_definitions( -DPROFILING_ON=1 ) #print profiling output

find_package(catkin REQUIRED roscpp roslib sensor_msgs cv_bridge urdf orocos_kdl kdl_parser rosbag message_filters)

find_package(Boost REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

#########
## GPU ##
#########
set(GLEW_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(CUDA)
#find_package(GLEW REQUIRED)
find_package(GLEW)
find_package(OpenGL REQUIRED)

include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OpenGL_INCLUDE_DIRS})
cuda_include_directories(${CUDA_CUT_INCLUDE_DIRS})

link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})

add_definitions(${OpenGL_DEFINITIONS})
add_definitions(${GLEW_DEFINITIONS})

if(NOT CUDA_FOUND)
    message(" CUDA not found!")
else(NOT CUDA_FOUND)
    message(" CUDA has been found")
endif(NOT CUDA_FOUND)

if(NOT OPENGL_FOUND)
    message(" OPENGL not found!")
else(NOT OPENGL_FOUND)
    message(" OPENGL has been found!")
endif(NOT OPENGL_FOUND)

if(NOT GLEW_FOUND)
    message(" GLEW not found!")
else(NOT GLEW_FOUND)
    message(" GLEW has been found!")
endif(NOT GLEW_FOUND)



# enable cuda debug information with -g -G -O0, to use with cuda-dbg
# use --ptxas-options=-v to see number of registers, local, shared and constant memory used in kernels
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -arch=sm_20)

##########################################
# Dealing with different assimp versions #
##########################################
set(ASSIMP_PREFIX_PATH ${ASSIMP_HOME} ${ENV_ASSIMP_HOME} /usr/local /usr/local/include /usr/local/lib /usr/include /usr/lib /usr/local/include/assimp /usr/include/assimp /usr/lib/assimp /usr/local/lib/assimp)

# a header only used in Version 3
set(FileToCheckV3 "cimport.h")
find_path(ASSIMP_INCLUDE_DIR_V3
  NAMES ${FileToCheckV3}
  PATHS ${ASSIMP_PREFIX_PATH}
)

# a header only used in Version 2
set(FileToCheckV2 "assimp.h")
find_path(ASSIMP_INCLUDE_DIR_V2
  NAMES ${FileToCheckV2}
  PATHS ${ASSIMP_PREFIX_PATH}
)

IF(EXISTS "${ASSIMP_INCLUDE_DIR_V3}/${FileToCheckV3}")
  message("-- ${FileToCheckV3} exists. Therefore Assimp V3 must be installed and will be used.")
  # add flag for ros distro
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_V3")
ELSEIF(EXISTS "${ASSIMP_INCLUDE_DIR_V2}/${FileToCheckV2}" )
  message("-- ${FileToCheckV2} exists. Therefore Assimp V2 must be installed and will be used.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_V2")
ELSE()
   message(FATAL_ERROR "-- Neither ${FileToCheckV2} nor ${FileToCheckV3} exists. Assimp doesn't seem to be installed.")
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    roscpp 
    roslib 
    sensor_msgs
    urdf
    orocos_kdl
    kdl_parser
    message_filters
  DEPENDS 
    OpenCV 
    eigen 
    assimp
)

###########
## Build ##
###########
include_directories(include)

set(state_filtering_headers

    # distribution
    include/${PROJECT_NAME}/distribution/distribution.hpp

    # distribution features
    include/${PROJECT_NAME}/distribution/features/unnormalized_evaluable.hpp
    include/${PROJECT_NAME}/distribution/features/evaluable.hpp
    include/${PROJECT_NAME}/distribution/features/sampleable.hpp
    include/${PROJECT_NAME}/distribution/features/gaussian_mappable.hpp
    include/${PROJECT_NAME}/distribution/features/moments_estimable.hpp
    include/${PROJECT_NAME}/distribution/features/moments_solvable.hpp

    # distribution implementations
    include/${PROJECT_NAME}/distribution/implementations/gaussian_distribution.hpp
    include/${PROJECT_NAME}/distribution/implementations/sum_of_deltas.hpp
    include/${PROJECT_NAME}/process_model/damped_brownian_motion.hpp
    include/${PROJECT_NAME}/process_model/integrated_damped_brownian_motion.hpp

    # filter interface
    include/${PROJECT_NAME}/filter/types.hpp
    include/${PROJECT_NAME}/filter/filter.hpp
    include/${PROJECT_NAME}/filter/filter_context.hpp

    # process model interface
    include/${PROJECT_NAME}/process_model/stationary_process_model.hpp

    # process model implementations
#    include/${PROJECT_NAME}/process_model/composed_stationary_process_model.hpp
    include/${PROJECT_NAME}/process_model/brownian_process_model.hpp
    include/${PROJECT_NAME}/process_model/occlusion_process_model.hpp

    # system state interface
    include/${PROJECT_NAME}/system_states/floating_body_system.hpp
    include/${PROJECT_NAME}/system_states/robot_kinematics.hpp
    include/${PROJECT_NAME}/system_states/rigid_body_system.hpp

    # measurement model interfaces
    include/${PROJECT_NAME}/observation_models/image_observation_model.hpp

    # measurement model specializations
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/cpu_image_observation_model.hpp
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/kinect_measurement_model.hpp
    include/${PROJECT_NAME}/observation_models/spkf/model_types.hpp
    include/${PROJECT_NAME}/observation_models/spkf/measurement_model.hpp
    include/${PROJECT_NAME}/observation_models/spkf/depth_measurement_model.hpp

    ## Particle specific implementations
#    include/${PROJECT_NAME}/filter/particle/particle_filter_context.hpp
    include/${PROJECT_NAME}/filter/particle/coordinate_filter.hpp

    ## SPKF specific interface
    include/${PROJECT_NAME}/filter/kalman/kalman_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/sp_kalman_filter_context.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/distribution_descriptor.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/euclidean_validation_gate.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ellipsoidal_validation_gate.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/types.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/sigma_point_matrix.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/sigma_point_transform.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/sigma_point_kalman_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/spkf_state_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/validation_gate_base.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/zero_validation_gate.hpp

    # Spkf specific implementations
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_transform.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/kappa_unscented_transform.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_kalman_filter_base.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_kalman_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/factorized_unscented_kalman_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/spkf/ukf/ukf_distribution_descriptor.hpp

    # tools
    include/${PROJECT_NAME}/tools/helper_functions.hpp
    include/${PROJECT_NAME}/tools/image_visualizer.hpp
    include/${PROJECT_NAME}/tools/cloud_visualizer.hpp
    include/${PROJECT_NAME}/tools/log_erfc.hpp
    include/${PROJECT_NAME}/tools/macros.hpp
    include/${PROJECT_NAME}/tools/object_file_reader.hpp
    include/${PROJECT_NAME}/tools/urdf_reader.hpp
    include/${PROJECT_NAME}/tools/part_mesh_model.hpp
    include/${PROJECT_NAME}/tools/pcl_interface.hpp
    include/${PROJECT_NAME}/tools/rigid_body_renderer.hpp
    include/${PROJECT_NAME}/tools/ros_interface.hpp
    include/${PROJECT_NAME}/tools/tracking_dataset.hpp


#    include/${PROJECT_NAME}/cpu_multi_object_tracker.hpp
)

set(state_filtering_sources
    src/${PROJECT_NAME}/observation_models/cpu_image_observation_model/cpu_image_observation_model.cpp

    ## Particle specific implementations
    src/${PROJECT_NAME}/filter/particle/coordinate_filter.cpp

    ## SPKF specific implementations
    src/${PROJECT_NAME}/filter/kalman/kalman_filter.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/sigma_point_matrix.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/euclidean_validation_gate.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ellipsoidal_validation_gate.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_transform.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ukf/kappa_unscented_transform.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_kalman_filter_base.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ukf/unscented_kalman_filter.cpp
    src/${PROJECT_NAME}/filter/kalman/spkf/ukf/factorized_unscented_kalman_filter.cpp

    src/${PROJECT_NAME}/tools/image_visualizer.cpp
    src/${PROJECT_NAME}/tools/cloud_visualizer.cpp
    src/${PROJECT_NAME}/tools/object_file_reader.cpp
    src/${PROJECT_NAME}/tools/urdf_reader.cpp
    src/${PROJECT_NAME}/tools/rigid_body_renderer.cpp
)


## state_filtering
message("CATKING LIBRARIES" ${catkin_LIBRARIES})

add_library(${PROJECT_NAME} ${state_filtering_headers} ${state_filtering_sources})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} assimp)

## multi_object_tracker
#add_executable(cpu_multi_object_tracker
#    src/${PROJECT_NAME}/cpu_multi_object_tracker_node.cpp)
#target_link_libraries(cpu_multi_object_tracker
#    ${PROJECT_NAME}
#    ${catkin_LIBRARIES}
#    ${OpenCV_LIBS}
#    ${PCL_LIBRARIES}
#    yaml-cpp)

#set(CUDA_FOUND)

if(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    message("building cuda implementation")

    set(state_filtering_gpu_HEADERS
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/gpu_image_observation_model.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_filter.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_multiple_filter.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/object_rasterizer.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/shader.hpp
        include/${PROJECT_NAME}/multi_object_tracker.hpp
#        include/${PROJECT_NAME}/robot_tracker.hpp
)


    set(state_filtering_gpu_SOURCES
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/gpu_image_observation_model.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_filter.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_multiple_filter.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/object_rasterizer.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/shader.cpp)

    add_library(${PROJECT_NAME}_gpu
        ${state_filtering_gpu_HEADERS}
        ${state_filtering_gpu_SOURCES})

    target_link_libraries(${PROJECT_NAME}_gpu
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})

    cuda_add_library(${PROJECT_NAME}_cuda
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_filter.cu)
    target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_cuda)

    ## multi_object_tracker
    add_executable(multi_object_tracker
        src/${PROJECT_NAME}/multi_object_tracker_node.cpp)
    target_link_libraries(multi_object_tracker
        ${PROJECT_NAME}
        ${PROJECT_NAME}_gpu
        ${PROJECT_NAME}_cuda
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
	${CUDA_LIBRARIES}
        yaml-cpp)



#    ## robot_tracker
#    add_executable(robot_tracker
#        src/${PROJECT_NAME}/robot_tracker_node.cpp)
#    target_link_libraries(robot_tracker
#        ${PROJECT_NAME}
#        ${PROJECT_NAME}_gpu
#        ${PROJECT_NAME}_cuda
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#	${CUDA_LIBRARIES}
#        yaml-cpp)
else(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    message(" Not building GPU filtering")
    ## robot_tracker
#    add_executable(robot_tracker
#      src/${PROJECT_NAME}/robot_tracker_node.cpp)
#    target_link_libraries(robot_tracker
#        ${PROJECT_NAME}
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#        yaml-cpp)
endif(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)

## Declare a cpp executable
# add_executable(state_filtering_node src/state_filtering_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(state_filtering_node state_filtering_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(state_filtering_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS state_filtering state_filtering_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_state_filtering.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
