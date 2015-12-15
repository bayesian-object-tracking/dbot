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
 * \file simple_shader_provider.cpp
 * \date Dec 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <string>

#include <dbot/util/simple_shader_provider.hpp>

namespace dbot
{
SimpleShaderProvider::SimpleShaderProvider(
    const std::string& fragment_shader_src,
    const std::string& vertex_shader_src,
    const std::string& geometry_shader_src)
    : fragment_shader_(fragment_shader_src),
      vertex_shader_(vertex_shader_src),
      geometry_shader_(geometry_shader_src)
{
}

std::string SimpleShaderProvider::fragment_shader() const
{
    return fragment_shader_;
}
std::string SimpleShaderProvider::vertex_shader() const
{
    return vertex_shader_;
}

std::string SimpleShaderProvider::geometry_shader() const
{
    return geometry_shader_;
}

bool SimpleShaderProvider::has_geometry_shader() const
{
    return !geometry_shader_.empty();
}
}
