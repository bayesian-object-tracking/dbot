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
 * \file shader_provider.hpp
 * \date Dec 2015
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <string>

namespace dbot
{
/**
 * \brief Represents the ShaderProvider interface
 */
class ShaderProvider
{
public:
    /**
     * \brief Returns the fragment shader source code
     */
    virtual std::string fragment_shader() const = 0;

    /**
     * \brief Returns the vertex shader source code
     */
    virtual std::string vertex_shader() const = 0;

    /**
     * \brief Returns the vertex shader source code
     */
    virtual std::string geometry_shader() const = 0;

    /**
     * \brief Returns whether the current shader provider provides a geometry
     *        shader definition
     */
    virtual bool has_geometry_shader() const = 0;
};
}
