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
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \date November 2015
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <GL/glew.h>

#include <dbot/model/observation/gpu/shader_provider.hpp>

GLuint LoadShaders(const std::shared_ptr<dbot::ShaderProvider> &shaderProvider);
GLuint CreateShader(GLenum eShaderType, const std::string& shaderCode);
GLuint CreateProgram(const std::vector<GLuint> &shaderList);
