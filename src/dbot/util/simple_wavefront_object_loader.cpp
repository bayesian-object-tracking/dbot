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
 * \file simple_wavefront_object_model_loader.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <dbot/util/simple_wavefront_object_loader.hpp>

namespace dbot
{
SimpleWavefrontObjectModelLoader::SimpleWavefrontObjectModelLoader(
    const ObjectResourceIdentifier& ori)
    : ori_(ori)
{
}

void SimpleWavefrontObjectModelLoader::load(
    std::vector<std::vector<Eigen::Vector3d>>& vertices,
    std::vector<std::vector<std::vector<int>>>& triangle_indices) const
{
    vertices.resize(ori_.count_meshes());
    triangle_indices.resize(ori_.count_meshes());

    for (size_t i = 0; i < ori_.count_meshes(); i++)
    {
        ObjectFileReader file_reader;
        file_reader.set_filename(ori_.mesh_path(i));
        file_reader.Read();

        vertices[i] = *file_reader.get_vertices();
        triangle_indices[i] = *file_reader.get_indices();
    }
}
}
