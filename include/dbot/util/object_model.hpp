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

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>

#include <fl/util/types.hpp>

#include <dbot/util/object_model_loader.hpp>

namespace dbot
{
class ObjectModel
{
public:
    typedef typename Eigen::Transform<fl::Real, 3, Eigen::Affine> Affine;

    typedef std::vector<std::vector<Eigen::Vector3d>> Vertices;
    typedef std::vector<std::vector<std::vector<int>>> TriangleIndecies;

    ObjectModel() = default;

    ObjectModel(const std::shared_ptr<ObjectModelLoader>& loader, bool center)
    {
        load_from(loader, center);
    }

    void load_from(const std::shared_ptr<ObjectModelLoader>& loader,
                     bool center)
    {
        loader->load(vertices_, triangle_indices_);
        compute_centers(centers_);

        if (center) center_vertices(centers_, vertices_);
    }

    const Vertices& vertices() const { return vertices_; }
    const TriangleIndecies& triangle_indices() const
    {
        return triangle_indices_;
    }

    const std::vector<Eigen::Vector3d>& centers() const { return centers_; }
private:
    void compute_centers(std::vector<Eigen::Vector3d>& centers)
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

    void center_vertices(const std::vector<Eigen::Vector3d>& centers,
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

private:
    std::vector<Eigen::Vector3d> centers_;
    std::vector<Affine> default_poses_;

    std::vector<std::vector<Eigen::Vector3d>> vertices_;
    std::vector<std::vector<std::vector<int>>> triangle_indices_;
};
}
