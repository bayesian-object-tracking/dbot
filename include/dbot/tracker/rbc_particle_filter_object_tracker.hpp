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
 * \file rbc_particle_filter_object_tracker.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <fl/model/process/interface/state_transition_function.hpp>

#include <dbot/tracker/object_tracker.hpp>
#include <dbot/rao_blackwell_coordinate_particle_filter.hpp>

namespace dbot
{
/**
 * \brief RbcParticleFilterObjectTracker
 */
class RbcParticleFilterObjectTracker : public ObjectTracker
{
public:
    typedef fl::StateTransitionFunction<State, Noise, Input> StateTransition;
    typedef RbObservationModel<State> ObservationModel;

    typedef RaoBlackwellCoordinateParticleFilter<StateTransition,
                                                 ObservationModel> Filter;

public:
    /**
     * \brief Creates the tracker
     *
     * \param filter
     *     Rbc particle filter instance
     * \param object_model
     *     Object model instance
     * \param camera_data
     *     Camera data container
     * \param update_rate
     *     Moving average update rate
     */
    RbcParticleFilterObjectTracker(
        const std::shared_ptr<Filter>& filter,
        const std::shared_ptr<ObjectModel>& object_model,
        const std::shared_ptr<CameraData>& camera_data,
        int evaluation_count,
        double update_rate);

    /**
     * \brief perform a single filter step
     *
     * \param image
     *     Current observation image
     */
    State on_track(const Obsrv& image);

    /**
     * \brief Initializes the particle filter with the given initial states and
     *    the number of evaluations
     * @param initial_states
     * @param evaluation_count
     */
    State on_initialize(const std::vector<State>& initial_states);

private:
    std::shared_ptr<Filter> filter_;
    int evaluation_count_;
};
}
