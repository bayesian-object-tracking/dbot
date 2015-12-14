##
## This is part of the Bayesian Object Tracking (bot),
## (https://github.com/bayesian-object-tracking)
##
## Copyright (c) 2015 Max Planck Society,
## 				 Autonomous Motion Department,
## 			     Institute for Intelligent Systems
##
## This Source Code Form is subject to the terms of the GNU General Public
## License License (GNU GPL). A copy of the license can be found in the LICENSE
## file distributed with this source code.
##

##
## Date November 2015
## Author Jan Issac (jan.issac@gmail.com)
##

include(ExternalProject)
include(CMakeParseArguments)

if(NOT catkin_FOUND)

    set(gtest_LIBRARY gtest_local)
    set(gtest_main_LIBRARY gtest_main_local)
    set(${PROJECT_NAME}_TEST_LIBS ${gtest_LIBRARY} ${gtest_main_LIBRARY})

    set(GTEST_FRAMEWORK gtest_framework)

    ExternalProject_Add(
        ${GTEST_FRAMEWORK}
        URL https://googletest.googlecode.com/files/gtest-1.6.0.zip
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
        INSTALL_COMMAND "" # do not install this library
        CMAKE_ARGS -Dgtest_disable_pthreads=ON -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    ExternalProject_Get_Property(${GTEST_FRAMEWORK} source_dir binary_dir)

    set(gtest_INCLUDE_DIR ${source_dir}/include)
    set(gtest_LIBRARY_PATH
            ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
    set(gtest_main_LIBRARY_PATH
            ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)

    add_library(${gtest_LIBRARY} STATIC IMPORTED GLOBAL)
    set_target_properties(${gtest_LIBRARY}
        PROPERTIES
        IMPORTED_LOCATION ${gtest_LIBRARY_PATH}
        IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
    add_dependencies(${gtest_LIBRARY} ${GTEST_FRAMEWORK})

    add_library(${gtest_main_LIBRARY} STATIC IMPORTED GLOBAL)
    set_target_properties(${gtest_main_LIBRARY}
        PROPERTIES
        IMPORTED_LOCATION ${gtest_main_LIBRARY_PATH}
        IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
    add_dependencies(${gtest_main_LIBRARY} ${gtest_LIBRARY})

    include_directories(${gtest_INCLUDE_DIR})

else(NOT catkin_FOUND)

    set(gtest_LIBRARY gtest)
    set(gtest_main_LIBRARY gtest_main)
    set(${PROJECT_NAME}_TEST_LIBS ${gtest_LIBRARY} ${gtest_main_LIBRARY})

endif(NOT catkin_FOUND)

function(${PROJECT_NAME}_add_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(${PROJECT_NAME}
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_NAME "${${PROJECT_NAME}_NAME}_test")

    if(NOT catkin_FOUND)
        add_executable(${TEST_NAME} ${${PROJECT_NAME}_SOURCES})
        target_link_libraries(${TEST_NAME}
            ${${PROJECT_NAME}_TEST_LIBS} ${${PROJECT_NAME}_LIBS})
        add_test(${TEST_NAME} ${TEST_NAME})
    else(NOT catkin_FOUND)
        catkin_add_gtest(${TEST_NAME} ${${PROJECT_NAME}_SOURCES})
        target_link_libraries(${TEST_NAME}
            ${${PROJECT_NAME}_TEST_LIBS} ${${PROJECT_NAME}_LIBS})
    endif(NOT catkin_FOUND)
endfunction(${PROJECT_NAME}_add_test)

