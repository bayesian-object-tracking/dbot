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
 * \file simple_shader_provider.hpp
 * \date Dec 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <string>

#include <dbot/model/observation/gpu/shader_provider.hpp>

namespace dbot
{
/**
 * \brief Represents a simple shader code provider which is simply set at
 *        creation
 */
class SimpleShaderProvider : public ShaderProvider
{
public:
    /**
     * \brief Creates a SimpleShaderProvider with static shader contents. The
     *        Geometry shader code is optional
     *
     * \param fragment_shader_src
     *          Required fragment shader code
     * \param vertex_shader_src
     *          Required vertex shader code
     * \param geometry_shader_src
     *          Optional geometry shader code
     */
    SimpleShaderProvider(const std::string& fragment_shader_src,
                         const std::string& vertex_shader_src,
                         const std::string& geometry_shader_src = "");

    /**
     * \brief Returns the fragment shader source code
     */
    std::string fragment_shader() const;

    /**
     * \brief Returns the vertex shader source code
     */
    std::string vertex_shader() const;

    /**
     * \brief Returns the geometry shader source code
     */
    std::string geometry_shader() const;

    /**
     * \brief Returns whether the current shader provider provides a geometry
     *        shader definition
     */
    bool has_geometry_shader() const;

protected:
    std::string fragment_shader_;
    std::string vertex_shader_;
    std::string geometry_shader_;
};
}
