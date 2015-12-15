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
 * \file file_shader_provider.hpp
 * \date Dec 2015
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <string>
#include <fstream>

#include <dbot/util/file_shader_provider.hpp>

namespace dbot
{
FileShaderProvider::FileShaderProvider(const std::string& fragment_shader_file,
                                       const std::string& vertex_shader_file,
                                       const std::string& geometry_shader_file)
    : SimpleShaderProvider(load_file_content(fragment_shader_file),
                           load_file_content(vertex_shader_file),
                           load_file_content(geometry_shader_file))
{
}

std::string FileShaderProvider::load_file_content(const std::string& file)
{
    std::string content;

    // if none is specified then return empty string
    if (file.compare("none") == 0) return content;

    std::ifstream shader_stream(file, std::ios::in);

    if (!shader_stream.is_open()) throw LoadingShaderFileFailedException();

    std::string Line = "";
    while (std::getline(shader_stream, Line)) content += "\n" + Line;
    shader_stream.close();

    return content;
}
}
