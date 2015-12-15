/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/**
 * \file simple_shader_provider_test.hpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date Dec 2015
 */

#include <gtest/gtest.h>

#include <dbot/util/simple_shader_provider.hpp>

TEST(SimpleShaderProviderTests, without_geometry_shaders)
{
    dbot::SimpleShaderProvider shader_provider("fragment shader code",
                                               "vertex shader code");

    EXPECT_EQ(shader_provider.fragment_shader().compare("fragment shader code"),
              0);

    EXPECT_EQ(shader_provider.vertex_shader().compare("vertex shader code"),
              0);

    EXPECT_TRUE(shader_provider.geometry_shader().empty());
    EXPECT_FALSE(shader_provider.has_geometry_shader());
}

TEST(SimpleShaderProviderTests, with_geometry_shaders)
{
    dbot::SimpleShaderProvider shader_provider("fragment shader code",
                                               "vertex shader code",
                                               "geometry shader code");

    EXPECT_EQ(shader_provider.fragment_shader().compare("fragment shader code"),
              0);

    EXPECT_EQ(shader_provider.vertex_shader().compare("vertex shader code"),
              0);

    EXPECT_EQ(shader_provider.geometry_shader().compare("geometry shader code"),
              0);

    EXPECT_TRUE(shader_provider.has_geometry_shader());
}
