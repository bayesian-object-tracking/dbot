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

#pragma once

#include <string>
#include <exception>

#include <dbot/simple_shader_provider.hpp>

namespace dbot
{
/**
 * \brief Represents an exception thrown if loading the shader file failed
 */
class LoadingShaderFileFailedException : public std::exception
{
};

/**
 * \brief Represents a shader provider loading the source code from files
 */
class FileShaderProvider : public SimpleShaderProvider

{
public:
    /**
     * \brief Creates a FileShaderProvider which loads the shader source code
     *        from the specified files. The geometry shader is optional.
     *
     * \param fragment_shader_file
     *          Required fragment shader file
     * \param vertex_shader_file
     *          Required vertex shader file
     * \param geometry_shader_file
     *          Optional geometry shader file. if no geometry shader is given,
     *          set the argument to "none"
     */
    FileShaderProvider(const std::string& fragment_shader_file,
                       const std::string& vertex_shader_file,
                       const std::string& geometry_shader_file = "none");

protected:
    /**
     * \brief Loads shader file content
     *
     * \param [in]    file
     *          Source file path
     * \param [out]  content
     *          Loaded file content
     *
     * \throws LoadingShaderFileFailedException
     */
    std::string load_file_content(const std::string& file);
};
}
