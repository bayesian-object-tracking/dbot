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
 * \file file_shader_provider_test.hpp
 * \author Jan Issc (jan.issac@gmail.com)
 * \date Dec 2015
 */

#include <gtest/gtest.h>

#include <fstream>
#include <boost/filesystem.hpp>

#include <dbot/file_shader_provider.hpp>

std::string write_temp_file(const std::string& content)
{
    boost::filesystem::path temp = boost::filesystem::unique_path();
    std::ofstream ofs;
    ofs.open(temp.c_str());
    ofs << content;
    ofs.close();

    return temp.string();
}

TEST(FileShaderProviderTests, without_geometry_shaders)
{
    std::string fragment_shader_file = write_temp_file("fragment shader code");
    std::string vertex_shader_file = write_temp_file("vertex shader code");

    dbot::FileShaderProvider shader_provider(fragment_shader_file,
                                             vertex_shader_file);

    EXPECT_EQ(shader_provider.fragment_shader().compare("fragment shader code"),
              0);

    EXPECT_EQ(shader_provider.vertex_shader().compare("vertex shader code"),
              0);

    EXPECT_TRUE(shader_provider.geometry_shader().empty());
    EXPECT_FALSE(shader_provider.has_geometry_shader());
}

TEST(FileShaderProviderTests, with_geometry_shaders)
{
    std::string fragment_shader_file = write_temp_file("fragment shader code");
    std::string vertex_shader_file = write_temp_file("vertex shader code");
    std::string geometry_shader_file = write_temp_file("geometry shader code");

    dbot::FileShaderProvider shader_provider(fragment_shader_file,
                                             vertex_shader_file,
                                             geometry_shader_file);

    EXPECT_EQ(shader_provider.fragment_shader().compare("fragment shader code"),
              0);

    EXPECT_EQ(shader_provider.vertex_shader().compare("vertex shader code"),
              0);

    EXPECT_EQ(shader_provider.geometry_shader().compare("geometry shader code"),
              0);

    EXPECT_TRUE(shader_provider.has_geometry_shader());
}

