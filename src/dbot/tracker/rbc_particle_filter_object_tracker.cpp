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

#include <dbot/tracker/rbc_particle_filter_object_tracker.hpp>

namespace dbot
{
RbcParticleFilterObjectTracker::RbcParticleFilterObjectTracker(
    const std::shared_ptr<Filter>& filter,
    const dbot::ObjectModel& object_model,
    const dbot::CameraData& camera_data)
    : filter_(filter),
      object_model_(object_model),
      camera_data_(camera_data)
{
}

void RbcParticleFilterObjectTracker::initialize(
    const std::vector<State>& initial_states,
    int evaluation_count)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<State> states;
    for (auto state : initial_states)
    {
        states.push_back(to_center_coordinate_system(state));
    }

    filter_->set_particles(states);
    filter_->filter(camera_data_.depth_image(), zero_input());
    filter_->resample(evaluation_count / object_model_.count_parts());

    State delta_mean = filter_->belief().mean();

    filter_->observation_model()->integrated_poses().apply_delta(delta_mean);

    for (size_t i = 0; i < filter_->belief().size(); i++)
    {
        filter_->belief().location(i).center_around_zero(delta_mean);

        // this needs to be set to zero, because as we switch coordinate
        // system, the linear velocity changes, since it has to account for
        // part of the angular velocity.
        filter_->belief().location(i).set_zero_velocity();
    }
}

auto RbcParticleFilterObjectTracker::track(const Obsrv& image) -> State
{
    std::lock_guard<std::mutex> lock(mutex_);

    filter_->filter(image, zero_input());

    State delta_mean = filter_->belief().mean();

    for (size_t i = 0; i < filter_->belief().size(); i++)
    {
        filter_->belief().location(i).center_around_zero(delta_mean);
    }

    auto& integrated_poses = filter_->observation_model()->integrated_poses();
    integrated_poses.apply_delta(delta_mean);

    return to_model_coordinate_system(integrated_poses);
}

auto RbcParticleFilterObjectTracker::to_center_coordinate_system(
    const RbcParticleFilterObjectTracker::State& state) -> State
{
    State centered_state = state;
    for (size_t j = 0; j < state.count(); j++)
    {
        centered_state.component(j).position() +=
            state.component(j).orientation().rotation_matrix() *
            object_model_.centers()[j];
    }

    return centered_state;
}

auto RbcParticleFilterObjectTracker::to_model_coordinate_system(
    const RbcParticleFilterObjectTracker::State& state) -> State
{
    State model_state = state;
    for (size_t j = 0; j < state.count(); j++)
    {
        model_state.component(j).position() -=
            state.component(j).orientation().rotation_matrix() *
            object_model_.centers()[j];
    }

    return model_state;
}

RbcParticleFilterObjectTracker::Input
RbcParticleFilterObjectTracker::zero_input() const
{
    return Input::Zero(object_model_.count_parts() * 6);
}
}
