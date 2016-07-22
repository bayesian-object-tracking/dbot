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

#include <dbot/tracker/rbc_particle_filter_object_tracker.hpp>

namespace dbot
{
RbcParticleFilterObjectTracker::RbcParticleFilterObjectTracker(
    const std::shared_ptr<Filter>& filter,
    const std::shared_ptr<ObjectModel>& object_model,
    int evaluation_count,
    double update_rate,
    bool center_object_frame)
    : ObjectTracker(object_model, update_rate, center_object_frame),
      filter_(filter),
      evaluation_count_(evaluation_count)
{
}

auto RbcParticleFilterObjectTracker::on_initialize(
    const std::vector<State>& initial_states) -> State
{
    filter_->set_particles(initial_states);
    filter_->resample(evaluation_count_ / object_model_->count_parts());

    State delta_mean = filter_->belief().mean();

    for (size_t i = 0; i < filter_->belief().size(); i++)
    {
        filter_->belief().location(i).center_around_zero(delta_mean);
    }

    auto& integrated_poses = filter_->sensor()->integrated_poses();
    integrated_poses.apply_delta(delta_mean);

    return integrated_poses;
}

auto RbcParticleFilterObjectTracker::on_track(const Obsrv& image) -> State
{
    filter_->filter(image, zero_input());

    State delta_mean = filter_->belief().mean();

    for (size_t i = 0; i < filter_->belief().size(); i++)
    {
        filter_->belief().location(i).center_around_zero(delta_mean);
    }

    auto& integrated_poses = filter_->sensor()->integrated_poses();
    integrated_poses.apply_delta(delta_mean);

    return integrated_poses;
}
}
