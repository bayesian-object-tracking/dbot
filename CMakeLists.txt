cmake_minimum_required(VERSION 2.8.3)
project(state_filtering)

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED roscpp roslib sensor_msgs)

find_package(Boost REQUIRED)

find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)
cuda_include_directories(${CUDA_CUT_INCLUDE_DIRS})
if(NOT CUDA_FOUND)
    message(" CUDA not found!")
endif(NOT CUDA_FOUND)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(" OPENGL not found!")
endif(NOT OPENGL_FOUND)

# gpu stuff ----------------------------------------------------------------------------------------
set(GLEW_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
add_definitions(${GLEW_DEFINITIONS})
if(NOT GLEW_FOUND)
    message(" GLEW not found!")
endif(NOT GLEW_FOUND)

# enable cuda debug information with -g -G -O0, to use with cuda-dbg
# use --ptxas-options=-v to see number of registers, local, shared and constant memory used in kernels
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -arch=sm_20)


link_directories(${PROJECT_SOURCE_DIR}/local_install/lib)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp roslib sensor_msgs
  DEPENDS OpenCV eigen
)

###########
## Build ##
###########
include_directories(include)

set(state_filtering_HEADERS
    include/${PROJECT_NAME}/distribution/distribution.hpp
    include/${PROJECT_NAME}/distribution/sampleable.hpp    
    include/${PROJECT_NAME}/distribution/probability_density_function.hpp
    include/${PROJECT_NAME}/distribution/empirical_moments.hpp
    include/${PROJECT_NAME}/distribution/analytical_moments.hpp

    include/${PROJECT_NAME}/distribution/gaussian/gaussian_mappable.hpp
    include/${PROJECT_NAME}/distribution/gaussian/gaussian_sampleable.hpp
    include/${PROJECT_NAME}/distribution/gaussian/gaussian_distribution.hpp    
    include/${PROJECT_NAME}/distribution/brownian/damped_brownian_motion.hpp    
    include/${PROJECT_NAME}/distribution/brownian/integrated_damped_brownian_motion.hpp

    include/${PROJECT_NAME}/filter/types.hpp
    include/${PROJECT_NAME}/filter/filter.hpp
    include/${PROJECT_NAME}/filter/estimate.hpp
    include/${PROJECT_NAME}/filter/filter_context.hpp
#    include/${PROJECT_NAME}/filter/state_distribution.hpp

    include/${PROJECT_NAME}/process_model/stationary_process_model.hpp
    include/${PROJECT_NAME}/process_model/composed_stationary_process_model.hpp
    include/${PROJECT_NAME}/process_model/brownian_process_model.hpp

    include/${PROJECT_NAME}/filter/particle/particle_filter_context.hpp
    include/${PROJECT_NAME}/filter/particle/coordinate_filter.hpp

    include/${PROJECT_NAME}/filter/kalman/kalman_filter.hpp
    include/${PROJECT_NAME}/filter/kalman/sp_kalman_filter_context.hpp

    include/${PROJECT_NAME}/tools/helper_functions.hpp
    include/${PROJECT_NAME}/tools/image_visualizer.hpp
    include/${PROJECT_NAME}/tools/log_erfc.hpp
    include/${PROJECT_NAME}/tools/macros.hpp
    include/${PROJECT_NAME}/tools/object_file_reader.hpp
    include/${PROJECT_NAME}/tools/pcl_interface.hpp
    include/${PROJECT_NAME}/tools/rigid_body_renderer.hpp
    include/${PROJECT_NAME}/tools/ros_interface.hpp

    include/${PROJECT_NAME}/system_states/full_rigid_body_system.hpp
    include/${PROJECT_NAME}/system_states/rigid_body_system.hpp

    include/${PROJECT_NAME}/observation_models/image_observation_model.hpp
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/cpu_image_observation_model.hpp
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/gaussian_pixel_observation_model.hpp
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/occlusion_process_model.hpp
    include/${PROJECT_NAME}/observation_models/cpu_image_observation_model/pixel_observation_model.hpp
)

set(state_filtering_SOURCES    
    src/${PROJECT_NAME}/tools/image_visualizer.cpp
    src/${PROJECT_NAME}/tools/object_file_reader.cpp
    src/${PROJECT_NAME}/tools/rigid_body_renderer.cpp

    src/${PROJECT_NAME}/observation_models/cpu_image_observation_model/cpu_image_observation_model.cpp
    src/${PROJECT_NAME}/observation_models/cpu_image_observation_model/gaussian_pixel_observation_model.cpp
    src/${PROJECT_NAME}/observation_models/cpu_image_observation_model/occlusion_process_model.cpp
    src/${PROJECT_NAME}/observation_models/cpu_image_observation_model/pixel_observation_model.cpp

#    src/${PROJECT_NAME}/filter/particle/particle_filter_context.cpp
    src/${PROJECT_NAME}/filter/particle/coordinate_filter.cpp

#    src/${PROJECT_NAME}/filter/kalman/kalman_filter.cpp
#    src/${PROJECT_NAME}/filter/kalman/kalman_filter_context.cpp
)


## state_filtering
add_library(${PROJECT_NAME} ${state_filtering_HEADERS} ${state_filtering_SOURCES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

## run_cpu_coordinate_filter
add_executable(run_cpu_coordinate_filter
    src/${PROJECT_NAME}/run_cpu_coordinate_filter.cpp)
target_link_libraries(run_cpu_coordinate_filter
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    yaml-cpp)

if(CUDA_FOUND AND OpenGL_FOUND AND GLEW_FOUND)
    set(state_filtering_gpu_HEADERS
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/gpu_image_observation_model.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_filter.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/cuda_opengl_multiple_filter.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/object_rasterizer.hpp
        include/${PROJECT_NAME}/observation_models/gpu_image_observation_model/shader.hpp)

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

    add_executable(run_gpu_coordinate_filter
        src/${PROJECT_NAME}/run_gpu_coordinate_filter.cpp)
    target_link_libraries(run_gpu_coordinate_filter
        ${PROJECT_NAME}_gpu
        ${PROJECT_NAME}_cuda
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        yaml-cpp)
else(CUDA_FOUND AND OpenGL_FOUND AND GLEW_FOUND)
    message(" Not building GPU filtering")
endif(CUDA_FOUND AND OpenGL_FOUND AND GLEW_FOUND)


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
