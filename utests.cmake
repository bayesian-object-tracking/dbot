
dbot_add_test(
    NAME 	  object_resource_identifier
    SOURCES src/dbot/common/object_resource_identifier_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME 	  simple_shader_provider_test
    SOURCES src/dbot/common/simple_shader_provider_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME    file_shader_provider_test
    SOURCES src/dbot/common/file_shader_provider_test.cpp
    LIBS	  ${dbot_LIBRARIES})
