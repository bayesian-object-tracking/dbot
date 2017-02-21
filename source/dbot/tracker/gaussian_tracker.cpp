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

#include <dbot/tracker/gaussian_tracker.h>

namespace dbot
{
GaussianTracker::GaussianTracker(
    const std::shared_ptr<Filter>& filter,
    const std::shared_ptr<ObjectModel>& object_model,
    double update_rate,
    bool center_object_frame)
    : Tracker(object_model, update_rate, center_object_frame),
      filter_(filter),
      belief_(filter_->create_belief())
{
}

auto GaussianTracker::on_initialize(
    const std::vector<State>& initial_states) -> State
{
    auto initial_cov = belief_.covariance();
    initial_cov.setZero();

    belief_.mean(initial_states[0]);
    belief_.covariance(initial_cov);

    return belief_.mean();
}

auto GaussianTracker::on_track(const Obsrv& obsrv) -> State
{
    // the following is approximately ok, but to be correct in a differential
    // geometry sense, the covariance matrix would also have to change due
    // to the parallel transport


    /// \todo: this transformation has to be checked and shoudl not be done here

    State old_pose = belief_.mean();
    filter_->sensor().local_sensor().body_model().nominal_pose(old_pose);

    State zero_pose = belief_.mean();
    zero_pose.set_zero_pose();
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
