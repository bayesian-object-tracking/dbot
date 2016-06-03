/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file shader.cpp
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date November 2015
 */

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>

#include <dbot/gpu/shader.hpp>

GLuint LoadShaders(const std::shared_ptr<dbot::ShaderProvider>& shaderProvider)
{
    std::vector<GLuint> shaderList;
    shaderList.push_back(
        CreateShader(GL_VERTEX_SHADER, shaderProvider->vertex_shader()));

    if (shaderProvider->has_geometry_shader())
    {
        shaderList.push_back(CreateShader(GL_GEOMETRY_SHADER,
                                          shaderProvider->geometry_shader()));
    }

    shaderList.push_back(
        CreateShader(GL_FRAGMENT_SHADER, shaderProvider->fragment_shader()));

    GLuint theProgram = CreateProgram(shaderList);

    std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);

    return theProgram;
}

// source:
// http://www.arcsynthesis.org/gltut/Basics/Tut01%20Making%20Shaders.html, Jason
// L. McKesson, 2012
GLuint CreateShader(GLenum eShaderType, const std::string& shaderCode)
{
    GLuint shader = glCreateShader(eShaderType);

    const char* strFileData = shaderCode.c_str();
    glShaderSource(shader, 1, &strFileData, NULL);

    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar* strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char* strShaderType = NULL;
        switch (eShaderType)
        {
            case GL_VERTEX_SHADER:
                strShaderType = "vertex";
                break;
            case GL_GEOMETRY_SHADER:
                strShaderType = "geometry";
                break;
            case GL_FRAGMENT_SHADER:
                strShaderType = "fragment";
                break;
        }

        fprintf(stderr,
                "Compile failure in %s shader:\n%s\n",
                strShaderType,
                strInfoLog);
        delete[] strInfoLog;
    }

    return shader;
}

// source:
// http://www.arcsynthesis.org/gltut/Basics/Tut01%20Making%20Shaders.html, Jason
// L. McKesson, 2012
GLuint CreateProgram(const std::vector<GLuint>& shaderList)
{
    GLuint program = glCreateProgram();

    for (size_t iLoop = 0; iLoop < shaderList.size(); iLoop++)
        glAttachShader(program, shaderList[iLoop]);

    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar* strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        delete[] strInfoLog;
    }

    for (size_t iLoop = 0; iLoop < shaderList.size(); iLoop++)
        glDetachShader(program, shaderList[iLoop]);

    return program;
}
