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
 * \file object_model.h
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Dense>

#include <fl/util/types.hpp>

#include <dbot/object_model_loader.h>

namespace dbot
{
class ObjectModel
{
public:
    typedef std::vector<std::vector<Eigen::Vector3d>> Vertices;
    typedef std::vector<std::vector<std::vector<int>>> TriangleIndecies;

public:
    ObjectModel() = default;

    ObjectModel(const std::shared_ptr<ObjectModelLoader>& loader, bool center);

    void load_from(const std::shared_ptr<ObjectModelLoader>& loader,
                   bool center);

    const Vertices& vertices() const;

    const TriangleIndecies& triangle_indices() const;

    const std::vector<Eigen::Vector3d>& centers() const;

    int count_parts() const;

private:
    void compute_centers(std::vector<Eigen::Vector3d>& centers);

    void center_vertices(const std::vector<Eigen::Vector3d>& centers,
                         std::vector<std::vector<Eigen::Vector3d>>& vertices);

private:
    std::vector<Eigen::Vector3d> centers_;

    std::vector<std::vector<Eigen::Vector3d>> vertices_;
    std::vector<std::vector<std::vector<int>>> triangle_indices_;
};
}
