cmake_minimum_required(VERSION 2.8.3)
project(dbot)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++0x -O3 -g3)
add_definitions(-DPROFILING_ON=1) #print profiling output

find_package(catkin REQUIRED
    robot_state_pub
    fl
)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc_dbot
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
find_package(ccache)

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
        dbot
    CATKIN_DEPENDS
        robot_state_pub
        fl
    DEPENDS
        OpenCV
        eigen
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} )

set(PROJECT_NAME "dbot")

file(GLOB_RECURSE
     library_headers
     include/${PROJECT_NAME}/*.hpp
     include/${PROJECT_NAME}/*.h)

set(library_sources
    src/${PROJECT_NAME}/utils/rigid_body_renderer.cpp
)

add_library(${PROJECT_NAME} ${library_headers} ${library_sources})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

###########
## Tests ##
###########
