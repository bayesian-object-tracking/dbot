
dbot_add_test(
    NAME 	object_resource_identifier
    SOURCES object_resource_identifier_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME 	simple_shader_provider_test
    SOURCES simple_shader_provider_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME    file_shader_provider_test
    SOURCES file_shader_provider_test.cpp
    LIBS	${dbot_LIBRARIES})
