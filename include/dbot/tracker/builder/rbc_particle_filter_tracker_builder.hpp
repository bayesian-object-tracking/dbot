
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

#include <dbot/tracker/rbc_particle_filter_object_tracker.hpp>

namespace dbot
{
class RbcParticleFilterTrackerBuilder
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

    typedef RbcParticleFilterObjectTracker::Parameters Parameters;

public:
    RbcParticleFilterTrackerBuilder(const Parameters& param,
                                    const std::vector<State>& initial_states,
                                    const dbot::CameraData& camera_data)
        : param_(param), camera_data_(camera_data)
    {
    }

    std::shared_ptr<RbcParticleFilterObjectTracker> build()
    {
        ObjectModel object_model;
        object_model.load_from(
            std::shared_ptr<ObjectModelLoader>(
                new SimpleWavefrontObjectModelLoader(param_.ori)),
            true);

        std::shared_ptr<ObservationModel> obsrv_model;
        if (!param_.use_gpu)
        {
            obsrv_model = RbObservationModelCpuBuilder<State>(
                              param_.obsrv, object_model, camera_data_)
                              .build();
        }
        else
        {
#ifdef BUILD_GPU
            obsrv_model = RbObservationModelGpuBuilder<State>(
                              param_.obsrv, object_model, camera_data_)
                              .build();
#else
            ROS_FATAL("Tracker has not been compiled with GPU support!");
            exit(1);
#endif
        }

        BrownianMotionModelBuilder<State, Input> process_builder(
            param_.process);
        std::shared_ptr<StateTransition> process = process_builder.build();

        auto filter = std::shared_ptr<Filter>(new Filter(
            process,
            obsrv_model,
            create_sampling_blocks(
                param_.ori.count_meshes(),
                process->noise_dimension() / param_.ori.count_meshes()),
            param_.max_kl_divergence));

        auto tracker = std::make_shared<RbcParticleFilterObjectTracker>(
            filter, param_, camera_data_, object_model);

        return tracker;
    }

private:
    std::vector<std::vector<size_t>> create_sampling_blocks(
        int blocks,
        int block_size) const
    {
        std::vector<std::vector<size_t>> sampling_blocks(
            param_.ori.count_meshes());
        for (int i = 0; i < blocks; ++i)
        {
            for (int k = 0; k < block_size; ++k)
            {
                sampling_blocks[i].push_back(i * block_size + k);
            }
        }

        return sampling_blocks;
    }

private:
    Parameters param_;
    dbot::CameraData camera_data_;
};

}
