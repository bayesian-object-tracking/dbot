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
 * \file rigid_body_state.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include "pose_velocity_vector.hpp"

namespace osr
{
template <int Dimension = -1>
class RigidBodiesState : public Eigen::Matrix<double, Dimension, 1>
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Dimension, 1> State;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector;

    typedef osr::PoseVelocityBlock<State> PoseVelocityBlock;

    // constructor and destructor
    RigidBodiesState() {}
    template <typename T>
    RigidBodiesState(const Eigen::MatrixBase<T>& state_vector)
    {
        *this = state_vector;
    }

    virtual ~RigidBodiesState() noexcept {}
    template <typename T>
    void operator=(const Eigen::MatrixBase<T>& state_vector)
    {
        *((State*)(this)) = state_vector;
    }

    virtual osr::PoseVelocityVector component(int index) const = 0;

    virtual int count() const = 0;
};
}
