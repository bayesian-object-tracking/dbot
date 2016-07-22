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
 * \file rbc_particle_filter_tracker_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <exception>

#include <dbot/object_model_loader.hpp>
#include <dbot/object_resource_identifier.hpp>
#include <dbot/tracker/rbc_particle_filter_object_tracker.hpp>
#include <dbot/builder/object_transition_builder.hpp>
#include <dbot/builder/rb_sensor_builder.h>

namespace dbot
{

/**
 * \brief Represents an Rbc Particle filter based tracker builder
 */
template <typename Tracker>
class RbcParticleFilterTrackerBuilder
{
public:
    typedef typename Tracker::State State;
    typedef typename Tracker::Noise Noise;
    typedef typename Tracker::Input Input;

    /* == Model Builder Interfaces ========================================== */
    typedef TransitionFunctionBuilder<State, Noise, Input>
        TransitionBuilder;
    typedef RbSensorBuilder<State> SensorBuilder;

    /* == Model Interfaces ================================================== */
    typedef fl::TransitionFunction<State, Noise, Input> Transition;
    typedef RbSensor<State> Sensor;
    typedef typename Sensor::Observation Obsrv;

    /* == Filter algorithm ================================================== */
    typedef RaoBlackwellCoordinateParticleFilter<Transition,
                                                 Sensor> Filter;

    /* == Tracker parameters ================================================ */
    struct Parameters
    {
        int evaluation_count;
        double moving_average_update_rate;
        double max_kl_divergence;
        bool center_object_frame;
    };

public:
    /**
     * \brief Creates a RbcParticleFilterTrackerBuilder
     * \param param			Builder and sub-builder parameters
     */
    RbcParticleFilterTrackerBuilder(
        const std::shared_ptr<TransitionBuilder>& transition_builder,
        const std::shared_ptr<SensorBuilder>& sensor_builder,
        const std::shared_ptr<ObjectModel>& object_model,
        const Parameters& params)
        : transition_builder_(transition_builder),
          sensor_builder_(sensor_builder),
          object_model_(object_model),
          params_(params)
    {
    }

    /**
     * \brief Builds the Rbc PF tracker
     */
    std::shared_ptr<RbcParticleFilterObjectTracker> build()
    {
        auto filter = create_filter(object_model_, params_.max_kl_divergence);

        auto tracker = std::make_shared<RbcParticleFilterObjectTracker>(
            filter,
            object_model_,
            params_.evaluation_count,
            params_.moving_average_update_rate,
            params_.center_object_frame);        

        return tracker;
    }

    /**
     * \brief Creates an instance of the Rbc particle filter
     *
     * \throws NoGpuSupportException if compile with DBOT_BUILD_GPU=OFF and
     *         attempting to build a tracker with GPU support
     */
    virtual std::shared_ptr<Filter> create_filter(
        const std::shared_ptr<ObjectModel>& object_model,
        double max_kl_divergence)
    {
        auto transition = transition_builder_->build();
        auto sensor = sensor_builder_->build();

        auto sampling_blocks =
            create_sampling_blocks(object_model->count_parts(),
                                   transition->noise_dimension() /
                                       object_model->count_parts());

        auto filter = std::shared_ptr<Filter>(new Filter(transition,
                                                         sensor,
                                                         sampling_blocks,
                                                         max_kl_divergence));
        return filter;
    }

    /**
     * \brief Creates a sampling block definition used by the coordinate
     *        particle filter
     *
     * \param blocks		Number of objects or object parts
     * \param block_size	State dimension of each part
     */
    virtual std::vector<std::vector<int>> create_sampling_blocks(
        int blocks,
        int block_size) const
    {
        std::vector<std::vector<int>> sampling_blocks(blocks);
        for (int i = 0; i < blocks; ++i)
        {
            for (int k = 0; k < block_size; ++k)
            {
                sampling_blocks[i].push_back(i * block_size + k);
            }
        }

        return sampling_blocks;
    }

protected:
    std::shared_ptr<TransitionBuilder> transition_builder_;
    std::shared_ptr<SensorBuilder> sensor_builder_;
    std::shared_ptr<ObjectModel> object_model_;
    Parameters params_;
};
}
