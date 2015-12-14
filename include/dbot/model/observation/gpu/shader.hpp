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
 * \file shader.hpp
 * \author Claudia Pfreundt <claudilein@gmail.com>
 * \date November 2015
 */

#pragma once

#include <vector>
#include <string>
#include <GL/glew.h>

GLuint LoadShaders(std::vector<const char *> shaderFilePaths);
GLuint CreateShader(GLenum eShaderType, const char * strShaderFile);
GLuint CreateProgram(const std::vector<GLuint> &shaderList);
