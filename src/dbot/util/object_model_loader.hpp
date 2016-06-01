/*
 * this is part of the bayesian object tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * copyright (c) 2015 max planck society,
 * 				 autonomous motion department,
 * 			     institute for intelligent systems
 *
 * this source code form is subject to the terms of the gnu general public
 * license license (gnu gpl). a copy of the license can be found in the license
 * file distributed with this source code.
 */

#pragma once

#include <Eigen/Dense>

#include <vector>

namespace dbot
{

class ObjectModelLoader
{
public:
    virtual void load(
        std::vector<std::vector<Eigen::Vector3d>>& vertices,
        std::vector<std::vector<std::vector<int>>>& triangle_indices) const = 0;
};
}
