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

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <memory>
#include <mutex>

#include <dbot/util/object_file_reader.hpp>

#include <dbot/util/camera_data.hpp>
#include <dbot/util/object_model.hpp>
#include <dbot/util/object_model_loader.hpp>
#include <dbot/util/object_resource_identifier.hpp>
#include <dbot/util/simple_wavefront_object_loader.hpp>
#include <dbot/rao_blackwell_coordinate_particle_filter.hpp>
#include <dbot/model/state_transition/brownian_object_motion_model.hpp>

#include <fl/model/process/linear_state_transition_model.hpp>
#include <fl/model/process/interface/state_transition_function.hpp>

#include <osr/pose_vector.hpp>
#include <osr/composed_vector.hpp>

namespace dbot
{
/**
 * \brief RbcParticleFilterObjectTracker
 */
class RbcParticleFilterObjectTracker
{
public:
    typedef osr::FreeFloatingRigidBodiesState<> State;
    typedef Eigen::VectorXd Input;

    typedef fl::StateTransitionFunction<State, State, Input> StateTransition;
    typedef RbObservationModel<State> ObservationModel;
    typedef typename ObservationModel::Observation Obsrv;

    typedef RBCoordinateParticleFilter<StateTransition, ObservationModel>
        Filter;

public:
    RbcParticleFilterObjectTracker(const std::shared_ptr<Filter>& filter,
                                   const ObjectModel& object_model,
                                   const CameraData& camera_data,
                                   double update_rate);

    State track(const Obsrv& image);

    void initialize(const std::vector<State>& initial_states,
                    int evaluation_count);

    State to_center_coordinate_system(const State& state);
    State to_model_coordinate_system(const State& state);

    const CameraData& camera_data() const { return camera_data_; }
    void move_average(const State& moving_average,
                      State &new_state,
                      double update_rate);

private:
    Input zero_input() const;

private:
    std::shared_ptr<Filter> filter_;
    ObjectModel object_model_;
    CameraData camera_data_;
    State moving_average_;
    double update_rate_;
    std::mutex mutex_;
};
}
