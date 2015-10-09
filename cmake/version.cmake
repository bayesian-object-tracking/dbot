
find_package(Git)

execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE PROJECT_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# update version in documentation config
#configure_file(
#    ${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
#    ${PROJECT_BINARY_DIR}/Doxyfile @ONLY)
