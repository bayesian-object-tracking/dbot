cmake_minimum_required(VERSION 2.8.3)
project(dbot)

############################
# Options                  #
############################
option(BUILD_DBOT_ON_GPU "Compile CUDA enabled trackers" ON)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-std=c++0x)
add_definitions(-DPROFILING_ON=1) #print profiling output

add_definitions(-Wall)
add_definitions(-Wno-unused-local-typedefs)
add_definitions(-Wno-deprecated-declarations)
add_definitions(-Wno-comment)
# for eigen-3.1.2
add_definitions(-Wno-deprecated-register)

############################
# Setup                    #
############################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(dbot_INCLUDE_DIRS "include")
set(dbot_LIBRARY ${PROJECT_NAME})
set(dbot_LIBRARY_GPU ${dbot_LIBRARY}_gpu)

############################
# Exports                  #
############################
set(dbot_FOUND 1 PARENT_SCOPE)
set(dbot_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include PARENT_SCOPE)
set(dbot_LIBRARIES ${dbot_LIBRARY} PARENT_SCOPE)
set(dbot_DEFINITIONS ${Eigen_DEFINITIONS} PARENT_SCOPE)
set(dbot_DEPENDS Eigen PARENT_SCOPE)

############################
# Dependencies             #
############################
if(NOT fl_FOUND)
    find_package(fl REQUIRED)
endif(NOT fl_FOUND)

if(NOT osr_FOUND)
    find_package(osr REQUIRED)
endif(NOT osr_FOUND)

find_package(Eigen REQUIRED)
find_package(Boost REQUIRED)

include_directories(${fl_INCLUDE_DIRS})
include_directories(${osr_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_definitions(${Eigen_DEFINITIONS})

if(BUILD_DBOT_ON_GPU)
    set(GLEW_DIR ${CMAKE_MODULE_PATH})

    find_package(CUDA REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(OpenGL REQUIRED)

    include_directories(${GLEW_INCLUDE_DIRS})
    include_directories(${OpenGL_INCLUDE_DIRS})
    cuda_include_directories(${CUDA_CUT_INCLUDE_DIRS})

    link_directories(${OpenGL_LIBRARY_DIRS})
    link_directories(${GLEW_LIBRARY_DIRS})

    add_definitions(${OpenGL_DEFINITIONS})
    add_definitions(${GLEW_DEFINITIONS})

    # enable cuda debug information with -g -G -O0, to use with cuda-dbg use
    # --ptxas-options=-v to see number of registers, local, shared and constant
    # memory used in kernels
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -arch=sm_20)

    add_definitions( -DBUILD_GPU=1 )

    # add GLEW, OpenGL and CUDA paths
    list(APPEND dbot_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
    list(APPEND dbot_INCLUDE_DIRS ${OpenGL_INCLUDE_DIRS})
    list(APPEND dbot_INCLUDE_DIRS ${CUDA_CUT_INCLUDE_DIRS})
    list(APPEND dbot_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

    # add gpu sub library
    list(APPEND dbot_LIBRARIES ${PROJECT_NAME}_gpu)
endif(BUILD_DBOT_ON_GPU)

############################
## dbot library            #
############################
include_directories(${dbot_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

set(dbot_SOURCE_DIR src/${PROJECT_NAME})
set(dbot_HEADER_DIR include/${PROJECT_NAME})

file(GLOB_RECURSE dbot_HEADERS
    ${dbot_HEADER_DIR}/*.hpp
    ${dbot_HEADER_DIR}/*.h)

# Build dbot library
set(dbot_SOURCES
   ${dbot_SOURCE_DIR}/utils/rigid_body_renderer.cpp)

add_library(${dbot_LIBRARY}
    ${dbot_HEADERS}
    ${dbot_SOURCES})

target_link_libraries(${dbot_LIBRARY}
    ${catkin_LIBRARIES})

# Build dbot GPU library
if(BUILD_DBOT_ON_GPU)
    set(dbot_SOURCE_DIR_GPU
        ${dbot_SOURCE_DIR}/models/observation_models/kinect_image_observation_model_gpu)

    cuda_add_library(${dbot_LIBRARY_GPU}
        ${dbot_SOURCE_DIR_GPU}/cuda_filter.cu
        ${dbot_SOURCE_DIR_GPU}/object_rasterizer.cpp
        ${dbot_SOURCE_DIR_GPU}/shader.cpp)

    target_link_libraries(${dbot_LIBRARY_GPU}
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})
endif(BUILD_DBOT_ON_GPU)
