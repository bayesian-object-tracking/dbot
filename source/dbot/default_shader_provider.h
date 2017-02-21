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
 * \file default_shader_provider.h
 * \date Dec 2015
 * \author Claudia Pfreundt (claudilein@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <string>

#include <dbot/simple_shader_provider.h>

namespace dbot
{
/**
 * \brief Represents a simple shader code provider which is simply set at
 *        creation
 */
class DefaultShaderProvider : public SimpleShaderProvider
{
public:
    /**
     * \brief Creates a default shader provider with compile time static shader
     *        code
     */
    DefaultShaderProvider();
};
}
