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
 * J. Issac, M. Wuthrich, C. Garcia Cifuentes, J. Bohg, S. Trimpe, S. Schaal
 * Depth-Based Object Tracking Using a Robust Gaussian Filter
 * IEEE Intl Conf on Robotics and Automation, 2016
 * http://arxiv.org/abs/1602.06157
 *
 */

#include <dbot/tracker/rms_gaussian_filter_object_tracker.hpp>

namespace dbot
{
RmsGaussianFilterObjectTracker::RmsGaussianFilterObjectTracker(
    const std::shared_ptr<Filter>& filter,
    const std::shared_ptr<ObjectModel>& object_model,
    double update_rate)
    : ObjectTracker(object_model, update_rate),
      filter_(filter),
      belief_(filter_->create_belief())
{
}

auto RmsGaussianFilterObjectTracker::on_initialize(
    const std::vector<State>& initial_states) -> State
{
    auto initial_cov = belief_.covariance();
    initial_cov.setZero();

    belief_.mean(initial_states[0]);
    belief_.covariance(initial_cov);

    return belief_.mean();
}

auto RmsGaussianFilterObjectTracker::on_track(const Obsrv& obsrv) -> State
{
    State old_pose = belief_.mean();
    filter_->obsrv_model().local_obsrv_model().body_model().nominal_pose(
        old_pose);

    State zero_pose = belief_.mean();
    zero_pose.component(0).set_zero_pose();
    belief_.mean(zero_pose);

    filter_->predict(belief_, zero_input(), belief_);
    filter_->update(belief_, obsrv, belief_);

    State delta_mean = belief_.mean();
    State new_pose = old_pose;

    new_pose.apply_delta(delta_mean);
    belief_.mean(new_pose);

    return belief_.mean();
}
}
