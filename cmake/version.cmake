find_package(Git)

execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE PROJECT_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
