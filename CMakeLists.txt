cmake_minimum_required(VERSION 2.8.3)
project(dbot)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++0x -O3 -g3)
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
    robot_state_pub
    image_transport
    fl
)

# find_package(Doxygen)
# 
# if(DOXYGEN_FOUND)
#     configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
#     add_custom_target(doc
#         ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Generating API documentation with Doxygen" VERBATIM)
# endif(DOXYGEN_FOUND)

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

###################################
## catkin specific configuration ##
###################################
catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        pose_tracking
        pose_tracking_interface
    CATKIN_DEPENDS
        roscpp
        roslib
        sensor_msgs
        message_filters
        robot_state_pub
        image_transport
        fl
    DEPENDS
        OpenCV
        eigen
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} )

set(PROJECT_NAME_TRACKING "pose_tracking")
set(PROJECT_NAME_TRACKING_INTERFACE "pose_tracking_interface")

file(GLOB_RECURSE tracking_headers include/${PROJECT_NAME_TRACKING}/*.hpp include/${PROJECT_NAME_TRACKING}/*.h)
file(GLOB_RECURSE tracking_interface_headers include/${PROJECT_NAME_TRACKING_INTERFACE}/*.hpp include/${PROJECT_NAME_TRACKING_INTERFACE}/*.h)


set(tracking_sources
    src/${PROJECT_NAME_TRACKING}/utils/rigid_body_renderer.cpp
)

set(tracking_interface_sources
#    src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/object_tracker.cpp
#    src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/robot_tracker.cpp

    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/image_visualizer.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/cloud_visualizer.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/object_file_reader.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/tracking_dataset.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/pcl_interface.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/ros_interface.cpp
    src/${PROJECT_NAME_TRACKING_INTERFACE}/utils/image_publisher.cpp

    # TESTING FUKF
#    src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/fukf_test_tracker.cpp
)


## pose_tracking
#message("CATKING LIBRARIES" ${catkin_LIBRARIES})

add_library(${PROJECT_NAME_TRACKING} ${tracking_headers}
                                     ${tracking_interface_headers}
                                     ${tracking_sources})

target_link_libraries(${PROJECT_NAME_TRACKING} ${catkin_LIBRARIES}
                                               ${OpenCV_LIBRARIES}
                                               assimp)

add_library(${PROJECT_NAME_TRACKING_INTERFACE} ${tracking_headers}
                                               ${tracking_interface_headers}
                                               ${tracking_interface_sources})

target_link_libraries(${PROJECT_NAME_TRACKING_INTERFACE} ${catkin_LIBRARIES}
                                                         ${OpenCV_LIBRARIES}
                                                         assimp)

    ## dev_test_tracker
    add_executable(dev_test_tracker
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_tracker_node.cpp
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_tracker/virtual_object.hpp
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_tracker/vector_hashing.hpp
    )

    target_link_libraries(dev_test_tracker
        ${PROJECT_NAME_TRACKING}
        ${PROJECT_NAME_TRACKING_INTERFACE}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        yaml-cpp)


#    ## fpgf_test_tracker
#    add_executable(fpgf_test_tracker
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/fpgf_test_tracker_node.cpp
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_tracker/virtual_object.hpp
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_tracker/vector_hashing.hpp
#    )

#    target_link_libraries(fpgf_test_tracker
#        ${PROJECT_NAME_TRACKING}
#        ${PROJECT_NAME_TRACKING_INTERFACE}
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#        ${CUDA_LIBRARIES}
#        yaml-cpp)

#    ## test_param_estimation_ukf
#    add_executable(test_param_estimation_ukf
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/test_param_estimation_ukf.hpp
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/test_param_estimation_ukf_node.cpp
#    )

#    target_link_libraries(test_param_estimation_ukf
#        ${PROJECT_NAME_TRACKING}
#        ${PROJECT_NAME_TRACKING_INTERFACE}
#        ${PROJECT_NAME_TRACKING}_gpu
#        ${PROJECT_NAME_TRACKING}_cuda
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#        ${CUDA_LIBRARIES}
#        yaml-cpp)

    ## test_param_estimation_ukf
    add_executable(test_param_estimation_fpf
        src/${PROJECT_NAME_TRACKING_INTERFACE}/fpf_test/depth_pixel_model.hpp
        src/${PROJECT_NAME_TRACKING_INTERFACE}/fpf_test/squared_feature_policy.hpp
        src/${PROJECT_NAME_TRACKING_INTERFACE}/fpf_test/test_param_estimation_fpf.hpp
        src/${PROJECT_NAME_TRACKING_INTERFACE}/fpf_test/test_param_estimation_fpf_node.cpp
    )

    target_link_libraries(test_param_estimation_fpf
        ${PROJECT_NAME_TRACKING}
        ${PROJECT_NAME_TRACKING_INTERFACE}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
        yaml-cpp)


#    ## test_example plots
#    add_executable(dev_test_plots
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/test_param_estimation_ukf.hpp
#        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/dev_test_plots_node.cpp
#    )

#    target_link_libraries(dev_test_plots
#        ${PROJECT_NAME_TRACKING}
#        ${PROJECT_NAME_TRACKING_INTERFACE}
#        ${PROJECT_NAME_TRACKING}_gpu
#        ${PROJECT_NAME_TRACKING}_cuda
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
#        ${CUDA_LIBRARIES}
#        yaml-cpp)

    ## object_tracker
    add_executable(object_tracker
        src/${PROJECT_NAME_TRACKING_INTERFACE}/trackers/object_tracker_node.cpp)
    target_link_libraries(object_tracker
        ${PROJECT_NAME_TRACKING}
        ${PROJECT_NAME_TRACKING_INTERFACE}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        yaml-cpp)

