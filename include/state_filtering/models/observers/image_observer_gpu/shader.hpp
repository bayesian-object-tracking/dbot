#ifndef STATE_FILTERING_MODELS_OBSERVERS_SHADER_HPP
#define STATE_FILTERING_MODELS_OBSERVERS_SHADER_HPP

#include <vector>
#include <string>
#include <GL/glew.h>

GLuint LoadShaders(std::vector<const char *> shaderFilePaths);
GLuint CreateShader(GLenum eShaderType, const char * strShaderFile);
GLuint CreateProgram(const std::vector<GLuint> &shaderList);

#endif
