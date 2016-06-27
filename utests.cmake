
dbot_add_test(
    NAME 	  object_resource_identifier
    SOURCES source/dbot/object_resource_identifier_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME 	  simple_shader_provider_test
    SOURCES source/dbot/simple_shader_provider_test.cpp
    LIBS    ${dbot_LIBRARIES})

dbot_add_test(
    NAME    file_shader_provider_test
    SOURCES source/dbot/file_shader_provider_test.cpp
    LIBS	  ${dbot_LIBRARIES})
