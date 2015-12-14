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
 * \file rao_blackwell_observation_model.hpp
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Core>

#include <fl/util/types.hpp>
#include <osr/pose_vector.hpp>
#include <osr/pose_velocity_vector.hpp>
#include <osr/composed_vector.hpp>

namespace dbot
{
/// \todo this observation model is now specific to rigid body rendering,
/// terminology should be adapted accordingly.
template <typename State_>
class RbObservationModel
{
public:
    typedef State_ State;
    typedef Eigen::Matrix<fl::Real, Eigen::Dynamic, Eigen::Dynamic> Observation;

    typedef Eigen::Array<State, -1, 1> StateArray;
    typedef Eigen::Array<fl::Real, -1, 1> RealArray;
    typedef Eigen::Array<int, -1, 1> IntArray;
    typedef Eigen::Matrix<fl::Real, -1, 1> RealVector;
    typedef State PoseArray;

public:
    /// constructor and destructor *********************************************
    RbObservationModel(const fl::Real& delta_time) : delta_time_(delta_time) {}
    virtual ~RbObservationModel() noexcept {}
    /// likelihood computation *************************************************
    virtual RealArray loglikes(const StateArray& deviations,
                               IntArray& indices,
                               const bool& update = false) = 0;

    /// accessors **************************************************************
    virtual void set_observation(const Observation& image) = 0;
    virtual PoseArray& integrated_poses() { return default_poses_; }
    virtual void reset() = 0;

protected:
    fl::Real delta_time_;
    PoseArray default_poses_;
};
}
