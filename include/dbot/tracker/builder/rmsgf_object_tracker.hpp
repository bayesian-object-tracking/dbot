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
 * \file rmsgf_object_tracker.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

#include <osr/pose_vector.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/multi_sensor_gaussian_filter.hpp>
#include <fl/filter/gaussian/robust_multi_sensor_gaussian_filter.hpp>


#include "filter_builder_multi_sensor.hpp"
#include "filter_builder_robust_multi_sensor.hpp"

namespace rmsgf
{

template <
    template <typename ...> class FilterClass,
    template <typename...> class TailModelClass,
    typename Quadrature,
    typename StateType
>
class RmsgfObjectTracker
{
public:
    typedef FilterBuilder<FilterClass, TailModelClass, Quadrature, StateType> Builder;

    typedef typename Builder::Filter Filter;

    typedef typename fl::Traits<Filter>::State State;
    typedef typename fl::Traits<Filter>::Input Input;
    typedef typename fl::Traits<Filter>::Obsrv Obsrv;
    typedef typename fl::Traits<Filter>::Belief Belief;

    typedef typename Builder::ObsrvModel::Noise ObsrvNoise;

    struct Parameter
    {
        Parameter(Args& args)
            : builder_param(args)
        {
        }


        typename Builder::Parameter builder_param;

        void print()
        {
            builder_param.print();
        }
    };

public:
    /**
     * \brief Creates RmsgfObjectTracker
     */
    RmsgfObjectTracker(
        const State initial_pose,
        const std::shared_ptr<dbot::RigidBodyRenderer>& renderer,
        const Parameter& param)
        : filter_(Builder().build_filter(renderer, param.builder_param)),
          belief_(filter_.create_belief())
    {
        auto initial_cov = belief_.covariance();
        initial_cov.setZero();
        belief_.mean(initial_pose);
        belief_.covariance(initial_cov);
    }

    const Belief& belief() { return belief_; }
    Filter& filter() { return filter_; }

public:
    /**
     * \brief Runs the filter time and measurement update once
     */
    void filter_once(const Eigen::VectorXd& y)
    {
        /// \todo: this has to be double checked, i think
        /// we might not be handling the rotations 100 percent correctly
        State old_pose = belief_.mean();
        Builder::set_nominal_pose(filter_, old_pose);

        State zero_pose = belief_.mean();
        zero_pose.set_zero();
        belief_.mean(zero_pose);

        filter_.predict(belief_, Input::Zero(), belief_);
        filter_.update(belief_, y, belief_);

        State new_pose = belief_.mean();
        new_pose.set_zero();
        new_pose.orientation() = belief_.mean().orientation() * old_pose.orientation();
        new_pose.position() = belief_.mean().position() + old_pose.position();
        belief_.mean(new_pose);
    }

public:
    Filter filter_;
    Belief belief_;
};

}
