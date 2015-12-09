/*************************************************************************
This software allows for filtering in high-dimensional observation and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/

#include <fl/util/profiling.hpp>

#include <ros/ros.h>
#include <ros/package.h>

#include <dbot/model/state_transition/orientation_transition_function.hpp>
#include <dbot/tracker/rbc_particle_filter_object_tracker.hpp>

namespace dbot
{
RbcParticleFilterObjectTracker::RbcParticleFilterObjectTracker(
    const std::shared_ptr<Filter>& filter,
    const Parameters& param,
//    const std::vector<State>& initial_states,
    const dbot::CameraData& camera_data,
    const dbot::ObjectModel& object_model)
    : filter_(filter), param_(param), camera_data_(camera_data), object_model_(object_model)

{
//    initialize(initial_states);
}

void RbcParticleFilterObjectTracker::initialize(
    const std::vector<State>& initial_states)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<State> states;
    for (auto state : initial_states)
    {
        to_center_coordinate_system(state);
        states.push_back(state);
    }

    std::vector<State> multi_body_samples(states.size());
    for (size_t i = 0; i < multi_body_samples.size(); i++)
    {
        multi_body_samples[i] = states[i];
    }

    filter_->set_particles(multi_body_samples);

    filter_->filter(
        camera_data_.depth_image(),
        StateTransition::Input::Zero(param_.ori.count_meshes() * 6));
    filter_->resample(param_.evaluation_count / param_.ori.count_meshes());

    /// convert to a differential reperesentation ******************************
    State mean = filter_->belief().mean();
    filter_->observation_model()->default_poses().recount(mean.count());
    for (size_t i = 0; i < mean.count(); i++)
    {
        auto pose = filter_->observation_model()->default_poses().component(i);
        auto delta = mean.component(i);
        pose.orientation() = delta.orientation() * pose.orientation();
        pose.position() = delta.position() + pose.position();
    }

    for (size_t i_part = 0; i_part < filter_->belief().size(); i_part++)
    {
        State& state = filter_->belief().location(i_part);
        for (size_t i_obj = 0; i_obj < mean.count(); i_obj++)
        {
            state.component(i_obj).position() =
                state.component(i_obj).position() -
                mean.component(i_obj).position();

            state.component(i_obj).orientation() =
                state.component(i_obj).orientation() *
                mean.component(i_obj).orientation().inverse();

            // this needs to be set to zero, because as we switch coordinate
            // system, the linear velocity changes, since it has to account for
            // part of the angular velocity.
            state.component(i_obj).linear_velocity() = Eigen::Vector3d::Zero();
            state.component(i_obj).angular_velocity() = Eigen::Vector3d::Zero();
        }
    }
}

void RbcParticleFilterObjectTracker::to_center_coordinate_system(
    RbcParticleFilterObjectTracker::State& state)
{
    for (size_t j = 0; j < state.count(); j++)
    {
        state.component(j).position() +=
            state.component(j).orientation().rotation_matrix() *
            object_model_.centers()[j];
    }
}

void RbcParticleFilterObjectTracker::to_model_coordinate_system(
    RbcParticleFilterObjectTracker::State& state)
{
    for (size_t j = 0; j < state.count(); j++)
    {
        state.component(j).position() -=
            state.component(j).orientation().rotation_matrix() *
            object_model_.centers()[j];
    }
}

auto RbcParticleFilterObjectTracker::track(const Obsrv& image) -> State
{
    std::lock_guard<std::mutex> lock(mutex_);

    /// filter *****************************************************************
    //    INIT_PROFILING;
    filter_->filter(
        image, StateTransition::Input::Zero(param_.ori.count_meshes() * 6));
    //    MEASURE("-----------------> total time for filtering");

    /// convert to a differential reperesentation ******************************
    State mean_delta = filter_->belief().mean();
    filter_->observation_model()->default_poses().recount(mean_delta.count());
    for (size_t i = 0; i < mean_delta.count(); i++)
    {
        auto pose = filter_->observation_model()->default_poses().component(i);
        auto delta = mean_delta.component(i);
        pose.orientation() = delta.orientation() * pose.orientation();
        pose.position() = delta.position() + pose.position();
    }

    for (size_t i_part = 0; i_part < filter_->belief().size(); i_part++)
    {
        State& state = filter_->belief().location(i_part);
        for (size_t i_obj = 0; i_obj < mean_delta.count(); i_obj++)
        {
            state.component(i_obj).position() -=
                mean_delta.component(i_obj).position();
            state.component(i_obj).orientation() -=
                mean_delta.component(i_obj).orientation();
        }
    }

    /// visualize the mean state ***********************************************
    State mean = filter_->belief().mean();
    for (size_t i = 0; i < mean.count(); i++)
    {
        auto pose_0 =
            filter_->observation_model()->default_poses().component(i);
        auto state = mean.component(i);

        state.position() = state.position() + pose_0.position();
        state.orientation() = state.orientation() * pose_0.orientation();
    }

    to_model_coordinate_system(mean);

    return mean;
}
}
