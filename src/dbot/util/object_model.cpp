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
 * \file object_model.cpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <dbot/util/object_model.hpp>

namespace dbot
{
ObjectModel::ObjectModel(const std::shared_ptr<ObjectModelLoader>& loader,
                         bool center)
{
    load_from(loader, center);
}

void ObjectModel::load_from(const std::shared_ptr<ObjectModelLoader>& loader,
                            bool center)
{
    loader->load(vertices_, triangle_indices_);
    compute_centers(centers_);

    if (center) center_vertices(centers_, vertices_);
}

auto ObjectModel::vertices() const -> const Vertices &
{
    return vertices_;
}

auto ObjectModel::triangle_indices() const -> const TriangleIndecies &
{
    return triangle_indices_;
}

const std::vector<Eigen::Vector3d>& ObjectModel::centers() const
{
    return centers_;
}

int ObjectModel::count_parts() const
{
    return vertices_.size();
}
void ObjectModel::compute_centers(std::vector<Eigen::Vector3d>& centers)
{
    centers.resize(vertices_.size());
    for (size_t i = 0; i < vertices_.size(); i++)
    {
        centers[i] = Eigen::Vector3d::Zero();
        for (size_t j = 0; j < vertices_[i].size(); j++)
        {
            centers[i] += vertices_[i][j];
        }
        centers[i] /= double(vertices_[i].size());
    }
}

void ObjectModel::center_vertices(
    const std::vector<Eigen::Vector3d>& centers,
    std::vector<std::vector<Eigen::Vector3d>>& vertices)
{
    for (size_t i = 0; i < vertices.size(); i++)
    {
        for (size_t j = 0; j < vertices[i].size(); j++)
        {
            vertices[i][j] -= centers[i];
        }
    }
}
}
