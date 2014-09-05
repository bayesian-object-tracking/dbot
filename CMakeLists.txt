cmake_minimum_required(VERSION 2.8.3)
project(state_filtering)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_definitions( -DPROFILING_ON=1 -std=c++0x ) #print profiling output

find_package(catkin REQUIRED
    roscpp
    roslib
    sensor_msgs
    cv_bridge
    urdf
    orocos_kdl
    kdl_parser
    rosbag
    message_filters
    robot_state_pub
    image_transport
)

find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

set(PROJECT_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${PROJECT_MODULE_PATH})

find_package(Boost REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)

set(ccache_DIR ${PROJECT_MODULE_PATH})
find_package(ccache REQUIRED)

if(CCACHE_FOUND)
    message(" ccache found. GCC will be launched via ccache.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else(CCACHE_FOUND)
    message(" ccache not found. ")
endif(CCACHE_FOUND)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

#########
## GPU ##
#########
set(GLEW_DIR ${PROJECT_MODULE_PATH})

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
    INCLUDE_DIRS
        include
    LIBRARIES
        ${PROJECT_NAME}
    CATKIN_DEPENDS
        roscpp
        roslib
        sensor_msgs
        urdf
        orocos_kdl
        kdl_parser
        message_filters
        robot_state_pub
        image_transport
    DEPENDS
        OpenCV
        eigen
        assimp
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} )

set(PROJECT_NAME_FILTERING "fast_filtering")
set(PROJECT_NAME_TRACKING "pose_tracking")
set(PROJECT_NAME_TRACKING_INTERFACE "pose_tracking_interface")


file(GLOB_RECURSE filtering_headers include/${PROJECT_NAME_FILTERING}/*.hpp include/${PROJECT_NAME_FILTERING}/*.h)
file(GLOB_RECURSE tracking_headers include/${PROJECT_NAME_TRACKING}/*.hpp include/${PROJECT_NAME_TRACKING}/*.h)
file(GLOB_RECURSE tracking_interface_headers include/${PROJECT_NAME_TRACKING_INTERFACE}/*.hpp include/${PROJECT_NAME_TRACKING_INTERFACE}/*.h)


set(tracking_sources
    src/${PROJECT_NAME_TRACKING}/utils/rigid_body_renderer.cpp

    ## TODO: THESE SHOULD BE IN A SEPAREATE PACKAGE CALLED TRACKING INTERFACE
    src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/object_tracker.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/robot_tracker.cpp

    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/image_visualizer.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/cloud_visualizer.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/object_file_reader.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/kinematics_from_urdf.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/tracking_dataset.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/pcl_interface.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/ros_interface.cpp
)


## state_filtering
#message("CATKING LIBRARIES" ${catkin_LIBRARIES})

add_library(${PROJECT_NAME_TRACKING} ${filtering_headers} ${tracking_headers} ${tracking_interface_headers} ${tracking_sources})
target_link_libraries(${PROJECT_NAME_TRACKING} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} assimp)


#set(CUDA_FOUND False)

if(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    add_definitions( -DBUILD_GPU=1 )
    message("building cuda implementation")

    set(tracking_gpu_SOURCES
        src/${PROJECT_NAME_TRACKING}/models/observation_models/kinect_image_observation_model_gpu/cuda_opengl_filter.cpp
        src/${PROJECT_NAME_TRACKING}/models/observation_models/kinect_image_observation_model_gpu/cuda_opengl_multiple_filter.cpp
        src/${PROJECT_NAME_TRACKING}/models/observation_models/kinect_image_observation_model_gpu/object_rasterizer.cpp
        src/${PROJECT_NAME_TRACKING}/models/observation_models/kinect_image_observation_model_gpu/shader.cpp)

    add_library(${PROJECT_NAME_TRACKING}_gpu
        ${tracking_gpu_SOURCES})

    target_link_libraries(${PROJECT_NAME_TRACKING}_gpu
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})

    cuda_add_library(${PROJECT_NAME_TRACKING}_cuda
        src/${PROJECT_NAME_TRACKING}/models/observation_models/kinect_image_observation_model_gpu/cuda_filter.cu)
    target_link_libraries(${PROJECT_NAME_TRACKING} ${PROJECT_NAME_TRACKING}_cuda)

    ## object_tracker
    add_executable(object_tracker
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/object_tracker_node.cpp)
    target_link_libraries(object_tracker
        ${PROJECT_NAME_TRACKING}
        ${PROJECT_NAME_TRACKING}_gpu
        ${PROJECT_NAME_TRACKING}_cuda
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
        yaml-cpp)

    ## robot_tracker
    add_executable(robot_tracker
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/robot_tracker_node.cpp)
    target_link_libraries(robot_tracker
        ${PROJECT_NAME_TRACKING}
        ${PROJECT_NAME_TRACKING}_gpu
        ${PROJECT_NAME_TRACKING}_cuda
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        yaml-cpp)
else(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    message(" Not building GPU filtering")
    ## robot_tracker
    add_executable(robot_tracker
      src/${PROJECT_NAME}/trackers/robot_tracker_node.cpp)
    target_link_libraries(robot_tracker
        ${PROJECT_NAME_TRACKING}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        yaml-cpp)

    ## object_tracker
    add_executable(object_tracker
        src/${PROJECT_NAME}/trackers/object_tracker_node.cpp)
    target_link_libraries(object_tracker
        ${PROJECT_NAME_TRACKING}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        yaml-cpp)
endif(CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)

