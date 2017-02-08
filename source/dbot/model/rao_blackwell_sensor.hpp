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

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
 * Probabilistic Object Tracking using a Range Camera
 * IEEE Intl Conf on Intelligent Robots and Systems, 2013
 * http://arxiv.org/abs/1505.00241
 *
 */

/**
 * \file rao_blackwell_sensor.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Core>

#include <fl/util/types.hpp>
#include <dbot/pose/pose_vector.hpp>
#include <dbot/pose/pose_velocity_vector.hpp>
#include <dbot/pose/composed_vector.hpp>
#include <dbot/pose/free_floating_rigid_bodies_state.hpp>

namespace dbot
{
/// \todo this observation model is now specific to rigid body rendering,
/// terminology should be adapted accordingly.
template <typename State_>
class RbSensor
{
public:
    typedef State_ State;
    typedef Eigen::Matrix<fl::Real, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef Eigen::Array<State, -1, 1> StateArray;
    typedef Eigen::Array<fl::Real, -1, 1> RealArray;
    typedef Eigen::Array<int, -1, 1> IntArray;
    typedef Eigen::Matrix<fl::Real, -1, 1> RealVector;
//    typedef State PoseArray;

    /// \todo: this should be a different type, only containing poses
    typedef osr::FreeFloatingRigidBodiesState<> PoseArray;

public:
    /// constructor and destructor *********************************************
    RbSensor(const fl::Real& delta_time) : delta_time_(delta_time) {}
    virtual ~RbSensor() noexcept {}
    /// likelihood computation *************************************************
    virtual RealArray loglikes(const StateArray& deviations,
                               IntArray& indices,
                               const bool& update = false) = 0;

    // compute the loglikelihoods without keeping track of the occulsions
    virtual RealArray loglikes(const StateArray& deviations)
    {
        reset();
        IntArray zero_indices = IntArray::Zero(deviations.size());
        return loglikes(deviations, zero_indices, false);
    }

    /// accessors **************************************************************
    virtual void set_observation(const Observation& image) = 0;
    virtual PoseArray& integrated_poses() { return default_poses_; }
    virtual void reset() = 0;

protected:
    fl::Real delta_time_;
    PoseArray default_poses_;
};
}
