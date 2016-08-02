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

#include <dbot/simple_wavefront_object_loader.hpp>
#include <dbot/builder/particle_tracker_builder.hpp>


//namespace dbot
//{
//RbcParticleFilterTrackerBuilder::RbcParticleFilterTrackerBuilder(
//    const std::shared_ptr<TransitionFunctionBuilder>&
//        transition_builder,
//    const CameraData& camera_data)
//    : transition_builder_(transition_builder),
//      camera_data_(camera_data)
//{
//}

//std::shared_ptr<ParticleTracker>
//RbcParticleFilterTrackerBuilder::build()

//auto RbcParticleFilterTrackerBuilder::create_filter(
//    const ObjectModel& object_model,
//    double max_kl_divergence) -> std::shared_ptr<Filter>

//auto RbcParticleFilterTrackerBuilder::create_object_transition() const
//    -> std::shared_ptr<Transition>


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
