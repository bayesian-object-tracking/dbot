
include(ExternalProject)
include(CMakeParseArguments)

if(TARGET gtest)

    set(gtest_LIBRARY gtest)
    set(gtest_main_LIBRARY gtest_main)
    set(${PROJECT_NAME}_TEST_LIBS ${gtest_LIBRARY} ${gtest_main_LIBRARY})

else(TARGET gtest)

    message("No gtest found. Downloading and building gtest framework ...")

    set(gtest_LIBRARY ${PROJECT_NAME}_gtest)
    set(gtest_main_LIBRARY ${PROJECT_NAME}_gtest_main)
    set(${PROJECT_NAME}_TEST_LIBS ${gtest_LIBRARY} ${gtest_main_LIBRARY})

    set(GTEST_FRAMEWORK ${PROJECT_NAME}_gtest_framework)

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

endif(TARGET gtest)

function(${PROJECT_NAME}_add_test)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(${PROJECT_NAME}
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_NAME "${${PROJECT_NAME}_NAME}_test")

    add_executable(${TEST_NAME} ${${PROJECT_NAME}_SOURCES})
    target_link_libraries(${TEST_NAME}
        ${${PROJECT_NAME}_TEST_LIBS} ${${PROJECT_NAME}_LIBS})
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction(${PROJECT_NAME}_add_test)
