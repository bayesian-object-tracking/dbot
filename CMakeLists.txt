cmake_minimum_required(VERSION 2.8.3)
project(dbot)

############################
# Options                  #
############################
option(DBOT_ON_CUDA "Compile CUDA enabled trackers" ON)

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
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

############################
# Dependencies            ##
############################
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
add_definitions(${Eigen_DEFINITIONS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

############################
# GPU                      #
############################
if(DBOT_ON_GPU)
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
endif(DBOT_ON_GPU)

############################
## catkin                  #
############################
find_package(catkin REQUIRED fl osr_catkin)

if(DBOT_ON_GPU)
    catkin_package(
        INCLUDE_DIRS
            include
            ${GLEW_INCLUDE_DIRS}
            ${OpenGL_INCLUDE_DIRS}
            ${CUDA_CUT_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
        LIBRARIES
            dbot_gpu
            dbot
        CATKIN_DEPENDS
            fl
            osr_catkin
        DEPENDS
            eigen
    )
else(DBOT_ON_GPU)
    catkin_package(
        INCLUDE_DIRS
            include
        LIBRARIES
            dbot
        CATKIN_DEPENDS
            fl
            osr_catkin
        DEPENDS
            eigen
    )
endif(DBOT_ON_GPU)

include_directories(include ${catkin_INCLUDE_DIRS})

############################
## dbot library            #
############################
file(GLOB_RECURSE dbot_headers include/${PROJECT_NAME}/*.hpp
                               include/${PROJECT_NAME}/*.h)
set(dbot_sources src/${PROJECT_NAME}/utils/rigid_body_renderer.cpp)
add_library(${PROJECT_NAME} ${dbot_headers} ${dbot_sources})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

if(DBOT_ON_GPU)
    set(GPU_IMPL_PATH
        src/${PROJECT_NAME}/models/observation_models/kinect_image_observation_model_gpu)

    cuda_add_library(${PROJECT_NAME}_gpu
        ${GPU_IMPL_PATH}/cuda_filter.cu
        ${GPU_IMPL_PATH}/object_rasterizer.cpp
        ${GPU_IMPL_PATH}/shader.cpp)

    target_link_libraries(${PROJECT_NAME}_gpu
        ${catkin_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLFW_LIBRARY}
        ${GLEW_LIBRARIES})
endif(DBOT_ON_GPU)















