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

#include <dbot/tracker/builder/rbc_particle_filter_tracker_builder.hpp>

namespace dbot
{
RbcParticleFilterTrackerBuilder::RbcParticleFilterTrackerBuilder(
    const Parameters& param,
    const CameraData& camera_data)
    : param_(param), camera_data_(camera_data)
{
}

std::shared_ptr<RbcParticleFilterObjectTracker>
RbcParticleFilterTrackerBuilder::build()
{
    auto object_model = create_object_model(param_.ori);
    auto filter = create_filter(object_model, param_.max_kl_divergence);

    auto tracker = std::make_shared<RbcParticleFilterObjectTracker>(
        filter, object_model, camera_data_);

    return tracker;
}

auto RbcParticleFilterTrackerBuilder::create_filter(
    const ObjectModel& object_model,
    double max_kl_divergence) -> std::shared_ptr<Filter>
{
    auto state_transition_model = create_state_transition_model(param_.process);

    auto obsrv_model = create_obsrv_model(
        param_.use_gpu, object_model, camera_data_, param_.obsrv);

    auto sampling_blocks = create_sampling_blocks(
        object_model.count_parts(),
        state_transition_model->noise_dimension() / object_model.count_parts());

    auto filter = std::shared_ptr<Filter>(new Filter(state_transition_model,
                                                     obsrv_model,
                                                     sampling_blocks,
                                                     max_kl_divergence));

    return filter;
}

auto RbcParticleFilterTrackerBuilder::create_state_transition_model(
    const BrownianMotionModelBuilder<State, Input>& param) const
    -> std::shared_ptr<StateTransition>
{
    BrownianMotionModelBuilder<State, Input> process_builder(param);
    std::shared_ptr<StateTransition> process = process_builder.build();

    return process;
}

auto RbcParticleFilterTrackerBuilder::create_obsrv_model(
    bool use_gpu,
    const ObjectModel& object_model,
    const CameraData& camera_data,
    const RbObservationModelBuilder<State>::Parameters& param) const
    -> std::shared_ptr<ObservationModel>
{
    std::shared_ptr<ObservationModel> obsrv_model;

    if (!use_gpu)
    {
        obsrv_model = RbObservationModelCpuBuilder<State>(
                          param, object_model, camera_data)
                          .build();
    }
    else
    {
#ifdef BUILD_GPU
        obsrv_model = RbObservationModelGpuBuilder<State>(
                          param, object_model, camera_data)
                          .build();
#else
        ROS_FATAL("Tracker has not been compiled with GPU support!");
        exit(1);
#endif
    }

    return obsrv_model;
}

ObjectModel RbcParticleFilterTrackerBuilder::create_object_model(
    const ObjectResourceIdentifier& ori) const
{
    ObjectModel object_model;

    object_model.load_from(std::shared_ptr<ObjectModelLoader>(
                               new SimpleWavefrontObjectModelLoader(ori)),
                           true);

    return object_model;
}

std::vector<std::vector<size_t>>
RbcParticleFilterTrackerBuilder::create_sampling_blocks(int blocks,
                                                        int block_size) const
{
    std::vector<std::vector<size_t>> sampling_blocks(param_.ori.count_meshes());
    for (int i = 0; i < blocks; ++i)
    {
        for (int k = 0; k < block_size; ++k)
        {
            sampling_blocks[i].push_back(i * block_size + k);
        }
    }

    return sampling_blocks;
}

}
