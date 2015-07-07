cmake_minimum_required(VERSION 2.8.3)
project(dbot)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++0x )
add_definitions(-DPROFILING_ON=1) #print profiling output
set(USE_CUDA True)

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
    image_transport
    fl
)


set(PROJECT_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${PROJECT_MODULE_PATH})

find_package(Boost REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)



find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

###################################
## catkin specific configuration ##
###################################

if(USE_CUDA AND CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
catkin_package(
    INCLUDE_DIRS
        include
        ${GLEW_INCLUDE_DIRS}
        ${OpenGL_INCLUDE_DIRS}
        ${CUDA_CUT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    LIBRARIES
        dbot_cuda
        dbot_gpu
        dbot
    CATKIN_DEPENDS
        roscpp
        roslib
        sensor_msgs
        urdf
        orocos_kdl
        kdl_parser
        message_filters
        image_transport
        fl
    DEPENDS
        eigen)
else(USE_CUDA AND CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        dbot
    CATKIN_DEPENDS
        roscpp
        roslib
        sensor_msgs
        urdf
        orocos_kdl
        kdl_parser
        message_filters
        image_transport
        fl
    DEPENDS
        eigen)
endif(USE_CUDA AND CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} )


## dbot #######################################################################
set(PROJECT_NAME_DBOT "dbot")

file(GLOB_RECURSE dbot_headers include/${PROJECT_NAME_DBOT}/*.hpp
                               include/${PROJECT_NAME_DBOT}/*.h)
set(dbot_sources src/${PROJECT_NAME_DBOT}/utils/rigid_body_renderer.cpp)


add_library(${PROJECT_NAME_DBOT}     ${dbot_headers}
                                     ${dbot_sources})

target_link_libraries(${PROJECT_NAME_DBOT} ${catkin_LIBRARIES})



if(USE_CUDA AND CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)
    add_definitions( -DBUILD_GPU=1 )
    message("building cuda implementation")

    set(dbot_gpu_sources
        src/${PROJECT_NAME_DBOT}/models/observation_models/kinect_image_observation_model_gpu/cuda_opengl_filter.cpp
        src/${PROJECT_NAME_DBOT}/models/observation_models/kinect_image_observation_model_gpu/cuda_opengl_multiple_filter.cpp
        src/${PROJECT_NAME_DBOT}/models/observation_models/kinect_image_observation_model_gpu/object_rasterizer.cpp
        src/${PROJECT_NAME_DBOT}/models/observation_models/kinect_image_observation_model_gpu/shader.cpp)

    add_library(${PROJECT_NAME_DBOT}_gpu
        ${dbot_gpu_sources})

    target_link_libraries(${PROJECT_NAME_DBOT}_gpu
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})

    cuda_add_library(${PROJECT_NAME_DBOT}_cuda
        src/${PROJECT_NAME_DBOT}/models/observation_models/kinect_image_observation_model_gpu/cuda_filter.cu)
    target_link_libraries(${PROJECT_NAME_DBOT} ${PROJECT_NAME_DBOT}_cuda)
endif(USE_CUDA AND CUDA_FOUND AND OPENGL_FOUND AND GLEW_FOUND)















