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

#include <dbot/tracker/builder/brownian_motion_model_builder.hpp>
#include <dbot/tracker/builder/rb_observation_model_cpu_builder.hpp>
#include <dbot/tracker/builder/rb_observation_model_gpu_builder.hpp>

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
    typedef dbot::RbObservationModel<State> ObservationModel;
    typedef typename ObservationModel::Observation Obsrv;

    typedef dbot::RBCoordinateParticleFilter<StateTransition, ObservationModel>
        Filter;

    typedef typename Eigen::Transform<fl::Real, 3, Eigen::Affine> Affine;

public:
    struct Parameters
    {
        dbot::ObjectResourceIdentifier ori;

        bool use_gpu;

        int evaluation_count;
        double max_kl_divergence;

        RbObservationModelBuilder<State>::Parameters obsrv;
        BrownianMotionModelBuilder<State, Input>::Parameters process;
    };

    RbcParticleFilterObjectTracker(const std::shared_ptr<Filter>& filter,
                                   const Parameters& param,
                                   const dbot::CameraData& camera_data,
                                   const dbot::ObjectModel& object_model);

    void initialize(const std::vector<State>& initial_states);

    State track(const Obsrv& image);

    void to_center_coordinate_system(State& state);
    void to_model_coordinate_system(State& state);

    const Parameters& param() { return param_; }
    const dbot::CameraData& camera_data() const { return camera_data_; }
private:
    std::mutex mutex_;
    std::shared_ptr<Filter> filter_;

    ObjectModel object_model_;

    std::vector<Affine> default_poses_;
    Parameters param_;
    dbot::CameraData camera_data_;
};
}
