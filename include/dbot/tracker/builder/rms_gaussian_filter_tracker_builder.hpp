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
 * \file rms_gaussian_filter_tracker_builder.hpp
 * \date December 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <exception>

#include <dbot/util/object_resource_identifier.hpp>
#include <dbot/tracker/rms_gaussian_filter_object_tracker.hpp>
#include <dbot/tracker/builder/object_transition_model_builder.hpp>
#include <dbot/tracker/builder/rb_observation_model_cpu_builder.hpp>

namespace dbot
{
/**
 * \brief Represents an Rbc Particle filter based tracker builder
 */
class RmsGaussianFilterTrackerBuilder
{
public:
    typedef RmsGaussianFilterObjectTracker::State State;
    typedef RmsGaussianFilterObjectTracker::Input Input;
    typedef RmsGaussianFilterObjectTracker::Noise Noise;
    typedef RmsGaussianFilterObjectTracker::Obsrv Obsrv;
    typedef RmsGaussianFilterObjectTracker::Filter Filter;
    typedef RmsGaussianFilterObjectTracker::Quadrature Quadrature;
    typedef RmsGaussianFilterObjectTracker::StateTransition StateTransition;
    typedef RmsGaussianFilterObjectTracker::ObservationModel ObservationModel;

    struct Parameters
    {
        double ut_alpha;
        double update_rate;

        struct Observation
        {
            double bg_depth;
            double fg_noise_std;
            double bg_noise_std;
            double tail_weight;
            double uniform_tail_min;
            double uniform_tail_max;
            int sensors;
        };

        ObjectResourceIdentifier ori;
        Observation observation;
        ObjectTransitionModelBuilder<State, Input>::Parameters
            object_transition;
    };

public:
    /**
     * \brief Creates a RbcParticleFilterTrackerBuilder
     * \param param			Builder and sub-builder parameters
     * \param camera_data	Tracker camera data object
     */
    RmsGaussianFilterTrackerBuilder(const Parameters& param,
                                    const CameraData& camera_data);

    /**
     * \brief Builds the Rbc PF tracker
     */
    std::shared_ptr<RmsGaussianFilterObjectTracker> build();

private:
    /**
     * \brief Creates an instance of the Rbc particle filter
     */
    std::shared_ptr<Filter> create_filter(const ObjectModel& object_model);

    /**
     * \brief Creates a Linear object transition function used in the
     *        filter
     */
    StateTransition create_object_transition_model(
        const ObjectTransitionModelBuilder<State, Input>::Parameters& param)
        const;

    /**
     * \brief Creates the Rbc particle filter observation model. This can either
     *        be CPU or GPU based
     *
     * \throws NoGpuSupportException if compile with DBOT_BUILD_GPU=OFF and
     *         attempting to build a tracker with GPU support
     */
    ObservationModel create_obsrv_model(
        const ObjectModel& object_model,
        const CameraData& camera_data,
        const Parameters::Observation& param) const;

    /**
     * \brief Creates an object model renderer
     */
    std::shared_ptr<RigidBodyRenderer> create_renderer(
        const ObjectModel& object_model) const;

    /**
     * \brief Loads and creates an object model represented by the specified
     *        resource identifier
     */
    ObjectModel create_object_model(const ObjectResourceIdentifier& ori) const;

private:
    Parameters param_;
    dbot::CameraData camera_data_;
};
}
