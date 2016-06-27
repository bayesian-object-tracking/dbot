/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 *                    Autonomous Motion Department,
 *                    Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file default_shader_provider.cpp
 * \date Dec 2015
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <string>

#include <dbot/default_shader_provider.hpp>

namespace dbot
{
DefaultShaderProvider::DefaultShaderProvider()
    : SimpleShaderProvider(
          /* fragment shader */
          "#version 330 core                                                \n"
          "                                                                 \n"
          "in float depth;                                                  \n"
          "layout (location = 0) out float log_likelihood;                  \n"
          "                                                                 \n"
          "void main() {                                                    \n"
          "   log_likelihood = -depth;                                      \n"
          "}                                                                \n",
          /* vertex shader */
          "#version 330                                                     \n"
          "                                                                 \n"
          " // tell OpenGL which buffer corresponds to which input          \n"
          "layout(location = 0) in vec3 vertexPosition_modelspace;          \n"
          "out float depth;                                                 \n"
          "uniform mat4 MV;                                                 \n"
          "uniform mat4 P;                                                  \n"
          "                                                                 \n"
          "void main() {                                                    \n"
          "    // makes it homogenous                                       \n"
          "    vec4 v = vec4(vertexPosition_modelspace, 1);                 \n"
          "    vec4 tmp_position  = MV * v;                                 \n"
          "    depth = tmp_position.z;                                      \n"
          "    gl_Position = P * tmp_position;                              \n"
          "}                                                                \n")
{
}
}
