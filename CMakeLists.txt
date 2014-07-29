cmake_minimum_required(VERSION 2.8.3)
project(state_filtering)

set(CMAKE_BUILD_TYPE Release)
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
find_package(GLEW)
find_package(OpenGL)

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

file(GLOB_RECURSE state_filtering_headers include/*.hpp include/*.h)

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
    src/${PROJECT_NAME}/tools/kinematics_from_urdf.cpp
    src/${PROJECT_NAME}/tools/rigid_body_renderer.cpp
)


## state_filtering
#message("CATKING LIBRARIES" ${catkin_LIBRARIES})

add_library(${PROJECT_NAME} ${state_filtering_headers} ${state_filtering_sources})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} assimp)

## multi_object_tracker
add_executable(cpu_multi_object_tracker
    src/${PROJECT_NAME}/cpu_multi_object_tracker_node.cpp)
target_link_libraries(cpu_multi_object_tracker
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    yaml-cpp)

if(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    message("building cuda implementation")

    set(state_filtering_gpu_SOURCES
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/gpu_image_observation_model.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_filter.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_multiple_filter.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/object_rasterizer.cpp
        src/${PROJECT_NAME}/observation_models/gpu_image_observation_model/shader.cpp)

    add_library(${PROJECT_NAME}_gpu
        ${state_filtering_headers}
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



    ## robot_tracker
    add_executable(robot_tracker
        src/${PROJECT_NAME}/robot_tracker_node.cpp)
    target_link_libraries(robot_tracker
        ${PROJECT_NAME}
        ${PROJECT_NAME}_gpu
        ${PROJECT_NAME}_cuda
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
	${CUDA_LIBRARIES}
        yaml-cpp)
else(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    message(" Not building GPU filtering")
    ## robot_tracker
    add_executable(robot_tracker
      src/${PROJECT_NAME}/robot_tracker_node.cpp)
    target_link_libraries(robot_tracker
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        yaml-cpp)
endif(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
