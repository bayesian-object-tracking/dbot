#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>

#include <dbot/model/observation/gpu/shader.hpp>

GLuint LoadShaders(std::vector<const char *> shaderFilePaths) {

    std::vector<GLuint> shaderList;
    int numberOfShaders = shaderFilePaths.size();

    if (numberOfShaders == 2) {
        shaderList.push_back(CreateShader(GL_VERTEX_SHADER, shaderFilePaths[0]));
        shaderList.push_back(CreateShader(GL_FRAGMENT_SHADER, shaderFilePaths[1]));
    } else if (numberOfShaders == 3) {
        shaderList.push_back(CreateShader(GL_VERTEX_SHADER, shaderFilePaths[0]));
        shaderList.push_back(CreateShader(GL_GEOMETRY_SHADER, shaderFilePaths[1]));
        shaderList.push_back(CreateShader(GL_FRAGMENT_SHADER, shaderFilePaths[2]));
    }

    GLuint theProgram = CreateProgram(shaderList);

    std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);

    return theProgram;
}


// source: http://www.arcsynthesis.org/gltut/Basics/Tut01%20Making%20Shaders.html, Jason L. McKesson, 2012
GLuint CreateShader(GLenum eShaderType, const char * strShaderFile)
{
    GLuint shader = glCreateShader(eShaderType);


    std::string shaderCode;
    std::ifstream shaderStream(strShaderFile, std::ios::in);
    if(shaderStream.is_open()){
        std::string Line = "";
        while(getline(shaderStream, Line))
            shaderCode += "\n" + Line;
        shaderStream.close();
    }

    const char *strFileData = shaderCode.c_str();
    glShaderSource(shader, 1, &strFileData, NULL);

    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char *strShaderType = NULL;
        switch(eShaderType)
        {
        case GL_VERTEX_SHADER: strShaderType = "vertex"; break;
        case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
        case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
        }

        fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
        delete[] strInfoLog;
    }

    return shader;
}



// source: http://www.arcsynthesis.org/gltut/Basics/Tut01%20Making%20Shaders.html, Jason L. McKesson, 2012
GLuint CreateProgram(const std::vector<GLuint> &shaderList)
{
    GLuint program = glCreateProgram();

    for(size_t iLoop = 0; iLoop < shaderList.size(); iLoop++)
        glAttachShader(program, shaderList[iLoop]);

    glLinkProgram(program);

    GLint status;
    glGetProgramiv (program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        delete[] strInfoLog;
    }

    for(size_t iLoop = 0; iLoop < shaderList.size(); iLoop++)
        glDetachShader(program, shaderList[iLoop]);

    return program;
}
