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

#include <dbot/util/simple_wavefront_object_loader.hpp>
#include <dbot/tracker/builder/rbc_particle_filter_tracker_builder.hpp>


//namespace dbot
//{
//RbcParticleFilterTrackerBuilder::RbcParticleFilterTrackerBuilder(
//    const std::shared_ptr<StateTransitionFunctionBuilder>&
//        state_transition_builder,
//    const CameraData& camera_data)
//    : state_transition_builder_(state_transition_builder),
//      camera_data_(camera_data)
//{
//}

//std::shared_ptr<RbcParticleFilterObjectTracker>
//RbcParticleFilterTrackerBuilder::build()

//auto RbcParticleFilterTrackerBuilder::create_filter(
//    const ObjectModel& object_model,
//    double max_kl_divergence) -> std::shared_ptr<Filter>

//auto RbcParticleFilterTrackerBuilder::create_object_transition_model() const
//    -> std::shared_ptr<StateTransition>


//ObjectModel RbcParticleFilterTrackerBuilder::create_object_model(
//    const ObjectResourceIdentifier& ori) const

//std::vector<std::vector<size_t>>
//RbcParticleFilterTrackerBuilder::create_sampling_blocks(int blocks,
//                                                        int block_size) const
//{
//    std::vector<std::vector<size_t>> sampling_blocks(params_.ori.count_meshes());
//    for (int i = 0; i < blocks; ++i)
//    {
//        for (int k = 0; k < block_size; ++k)
//        {
//            sampling_blocks[i].push_back(i * block_size + k);
//        }
//    }

//    return sampling_blocks;
//}
//}
