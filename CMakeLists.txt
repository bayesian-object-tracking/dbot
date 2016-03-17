cmake_minimum_required(VERSION 2.8.3)
project(dbot)

############################
# Options                  #
############################
option(DBOT_BUILD_GPU "Compile CUDA enabled trackers" ON)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.6 or greater required
add_definitions(-std=c++0x -fno-omit-frame-pointer)
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
# include(cmake/version.cmake)

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

include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_definitions(${Eigen_DEFINITIONS})

# GPU libs
set(GLEW_DIR ${CMAKE_MODULE_PATH})
find_package(CUDA QUIET)
find_package(GLEW QUIET)
find_package(OpenGL QUIET)

if(DBOT_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)
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

  # activate gpu implementations
  add_definitions(-DDBOT_BUILD_GPU=1)
  set(DBOT_GPU_SUPPORT "YES")
else(DBOT_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)
  set(DBOT_GPU_SUPPORT "NO")
  set(DBOT_BUILD_GPU OFF)
endif(DBOT_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)

############################
# Library info summary     #
############################
#include(cmake/info.cmake)

#info_yesorno(CUDA_FOUND DBOT_FOUND_CUDA)
#info_yesorno(GLEW_FOUND DBOT_FOUND_GLEW)
#info_yesorno(OPENGL_FOUND DBOT_FOUND_OPENGL)

#info_begin()
#  info_project("::bot::dbot:: Depth Based Object Tracking Lib" ${PROJECT_VERSION})
#  info_header("Setup")
#  info_item("Found CUDA" "${DBOT_FOUND_CUDA}")
#  info_item("Found GLEW" "${DBOT_FOUND_GLEW}")
#  info_item("Found OpenGL" "${DBOT_FOUND_OPENGL}")
#  info_item("Compiling with GPU Support" "${DBOT_GPU_SUPPORT}")
#info_end()

############################
# Dependencies             #
############################
# Use catkin of available otherwise fall back to native cmake
find_package(catkin QUIET COMPONENTS fl osr)

set(DBOT_INCLUDE_DIRS "")
if(DBOT_BUILD_GPU)
  list(APPEND DBOT_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  list(APPEND DBOT_INCLUDE_DIRS ${CUDA_CUT_INCLUDE_DIRS})
  list(APPEND DBOT_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
  list(APPEND DBOT_INCLUDE_DIRS ${OpenGL_INCLUDE_DIR})
endif(DBOT_BUILD_GPU)

if(catkin_FOUND)
  message(STATUS "Using catkin")
  catkin_package(
    INCLUDE_DIRS
      include
      ${DBOT_INCLUDE_DIRS}
    LIBRARIES
      ${dbot_LIBRARIES}
    CATKIN_DEPENDS
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
    ${dbot_SOURCE_DIR}/util/object_model.cpp
    ${dbot_SOURCE_DIR}/util/object_file_reader.cpp
    ${dbot_SOURCE_DIR}/util/rigid_body_renderer.cpp
    ${dbot_SOURCE_DIR}/util/object_resource_identifier.cpp
    ${dbot_SOURCE_DIR}/util/simple_camera_data_provider.cpp
    ${dbot_SOURCE_DIR}/util/virtual_camera_data_provider.cpp
    ${dbot_SOURCE_DIR}/util/simple_wavefront_object_loader.cpp
    ${dbot_SOURCE_DIR}/util/simple_shader_provider.cpp
    ${dbot_SOURCE_DIR}/util/default_shader_provider.cpp
    ${dbot_SOURCE_DIR}/util/file_shader_provider.cpp
    ${dbot_SOURCE_DIR}/tracker/object_tracker.cpp
    ${dbot_SOURCE_DIR}/tracker/rbc_particle_filter_object_tracker.cpp
    ${dbot_SOURCE_DIR}/tracker/rms_gaussian_filter_object_tracker.cpp
    ${dbot_SOURCE_DIR}/tracker/builder/rb_observation_model_builder.cpp
    ${dbot_SOURCE_DIR}/tracker/builder/rbc_particle_filter_tracker_builder.cpp
    ${dbot_SOURCE_DIR}/tracker/builder/rms_gaussian_filter_tracker_builder.cpp
)

add_library(${dbot_LIBRARY} SHARED
    ${dbot_HEADERS}
    ${dbot_SOURCES})

target_link_libraries(${dbot_LIBRARY}
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES})

# Build dbot GPU library
if(DBOT_BUILD_GPU)
    set(dbot_SOURCE_DIR_GPU
        ${dbot_SOURCE_DIR}/model/observation/gpu)

    cuda_add_library(${dbot_LIBRARY_GPU} SHARED
        ${dbot_SOURCE_DIR}/model/observation/gpu/cuda_likelihood_evaluator.cu
        ${dbot_SOURCE_DIR}/model/observation/gpu/shader.cpp
        ${dbot_SOURCE_DIR}/model/observation/gpu/object_rasterizer.cpp
        ${dbot_SOURCE_DIR}/model/observation/gpu/buffer_configuration.cpp)

    target_link_libraries(${dbot_LIBRARY_GPU}
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})
endif(DBOT_BUILD_GPU)

############################
# Tests                    #
############################
enable_testing()
include(${CMAKE_MODULE_PATH}/gtest.cmake)
add_subdirectory(tests)
