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
add_definitions(-DPROFILING_ON=1 -fPIC) #print profiling output

#add_definitions(-Wall)
#add_definitions(-Wno-unused-local-typedefs)
#add_definitions(-Wno-deprecated-declarations)
#add_definitions(-Wno-comment)
## for eigen-3.1.2
#add_definitions(-Wno-deprecated-register)

############################
# Library Version          #
############################
include(cmake/version.cmake)

##########################
# Library info summary     #
############################
include(cmake/info.cmake)

info_begin()
  info_project("::bot::dbot:: Depth Based Object Tracking Lib" ${PROJECT_VERSION})
info_end()

############################
# Setup                    #
############################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# local variables
set(dbot_LIBRARY ${PROJECT_NAME})
set(dbot_LIBRARY_GPU ${dbot_LIBRARY}_gpu)

# parent scope variables; exported at the end
set(dbot_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(dbot_LIBRARIES ${dbot_LIBRARY})

############################
# Dependencies             #
############################

find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# include_directories(${fl_INCLUDE_DIRS})
# include_directories(${osr_INCLUDE_DIRS})
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
  list(APPEND dbot_LIBRARIES ${dbot_LIBRARY_GPU})

  add_definitions( -DBUILD_GPU=1 )
endif(BUILD_DBOT_ON_GPU)

############################
# Dependencies             #
############################
# Use catkin of available otherwise fall back to native cmake
find_package(catkin QUIET COMPONENTS fl osr roscpp roslib)
if(catkin_FOUND)
  message(STATUS "Using catkin")
  catkin_package(
    INCLUDE_DIRS
      include
      ${OpenGL_INCLUDE_DIRS}
      ${GLEW_INCLUDE_DIRS}
      ${CUDA_INCLUDE_DIRS}
      ${CUDA_CUT_INCLUDE_DIRS}
    LIBRARIES
      ${dbot_LIBRARIES}
    CATKIN_DEPENDS
      roscpp
      roslib
      osr
      fl
    DEPENDS
      Eigen)
  list(APPEND dbot_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})
else(catkin_FOUND)
  find_package(fl REQUIRED)
  find_package(osr REQUIRED)

  list(APPEND dbot_INCLUDE_DIRS ${fl_INCLUDE_DIRS})
  list(APPEND dbot_INCLUDE_DIRS ${osr_INCLUDE_DIRS})
endif(catkin_FOUND)

############################
## dbot library            #
############################
include_directories(${dbot_INCLUDE_DIRS})

set(dbot_SOURCE_DIR src/${PROJECT_NAME})
set(dbot_HEADER_DIR include/${PROJECT_NAME})

file(GLOB_RECURSE dbot_HEADERS
    ${dbot_HEADER_DIR}/*.hpp
    ${dbot_HEADER_DIR}/*.h)

# Build dbot library
set(dbot_SOURCES
    ${dbot_SOURCE_DIR}/util/camera_data.cpp
    ${dbot_SOURCE_DIR}/util/object_file_reader.cpp
    ${dbot_SOURCE_DIR}/util/rigid_body_renderer.cpp
    ${dbot_SOURCE_DIR}/util/object_resource_identifier.cpp
    ${dbot_SOURCE_DIR}/tracker/rbc_particle_filter_object_tracker.cpp)

add_library(${dbot_LIBRARY} SHARED
    ${dbot_HEADERS}
    ${dbot_SOURCES})

target_link_libraries(${dbot_LIBRARY}
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES})

# Build dbot GPU library
if(BUILD_DBOT_ON_GPU)
    set(dbot_SOURCE_DIR_GPU
        ${dbot_SOURCE_DIR}/model/observation/gpu)

    cuda_add_library(${dbot_LIBRARY_GPU} SHARED
        ${dbot_SOURCE_DIR}/model/observation/gpu/cuda_filter.cu
        ${dbot_SOURCE_DIR}/model/observation/gpu/shader.cpp
        ${dbot_SOURCE_DIR}/model/observation/gpu/object_rasterizer.cpp)

    target_link_libraries(${dbot_LIBRARY_GPU}
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})
endif(BUILD_DBOT_ON_GPU)

############################
# Tests                    #
############################
enable_testing()
include(${CMAKE_MODULE_PATH}/gtest.cmake)
add_subdirectory(tests)
