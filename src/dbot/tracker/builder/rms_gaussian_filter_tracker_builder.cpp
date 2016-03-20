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

#include <dbot/util/simple_wavefront_object_loader.hpp>
#include <dbot/tracker/builder/rms_gaussian_filter_tracker_builder.hpp>

namespace dbot
{
RmsGaussianFilterTrackerBuilder::RmsGaussianFilterTrackerBuilder(
    const Parameters& param,
    const std::shared_ptr<CameraData>& camera_data)
    : param_(param), camera_data_(camera_data)
{
}

std::shared_ptr<RmsGaussianFilterObjectTracker>
RmsGaussianFilterTrackerBuilder::build()
{
    auto object_model = create_object_model(param_.ori);

    auto filter = create_filter(object_model);

    auto tracker = std::make_shared<RmsGaussianFilterObjectTracker>(
        filter, object_model, camera_data_, param_.update_rate);

    return tracker;
}

auto RmsGaussianFilterTrackerBuilder::create_filter(
    const std::shared_ptr<ObjectModel>& object_model) -> std::shared_ptr<Filter>
{
    /* ------------------------------ */
    /* - State transition model     - */
    /* ------------------------------ */
    auto state_transition_model =
        create_object_transition_model(param_.object_transition);

    /* ------------------------------ */
    /* - Observation model          - */
    /* ------------------------------ */
    auto obsrv_model =
        create_obsrv_model(object_model, camera_data_, param_.observation);

    /* ------------------------------ */
    /* - Quadrature                 - */
    /* ------------------------------ */
    auto quadrature = Quadrature(param_.ut_alpha);

    /* ------------------------------ */
    /* - Filter                     - */
    /* ------------------------------ */
    auto filter = std::shared_ptr<Filter>(
        new Filter(state_transition_model, obsrv_model, quadrature));

    return filter;
}

auto RmsGaussianFilterTrackerBuilder::create_obsrv_model(
    const std::shared_ptr<ObjectModel>& object_model,
    const std::shared_ptr<CameraData>& camera_data,
    const Parameters::Observation& param) const -> ObservationModel
{
    typedef RmsGaussianFilterObjectTracker::PixelModel PixelModel;
    typedef RmsGaussianFilterObjectTracker::TailModel TailModel;
    typedef RmsGaussianFilterObjectTracker::BodyTailPixelModel BodyTailModel;

    auto renderer = create_renderer(object_model);

    auto pixel_obsrv_model = PixelModel(
        renderer, param.bg_depth, param.fg_noise_std, param.bg_noise_std);

    auto tail_obsrv_model =
        TailModel(param.uniform_tail_min, param.uniform_tail_max);

    auto body_tail_pixel_model =
        BodyTailModel(pixel_obsrv_model, tail_obsrv_model, param.tail_weight);

    return ObservationModel(body_tail_pixel_model, camera_data->pixels());
}

std::shared_ptr<ObjectModel>
RmsGaussianFilterTrackerBuilder::create_object_model(
    const ObjectResourceIdentifier& ori) const
{
    auto object_model = std::make_shared<ObjectModel>(
        std::shared_ptr<ObjectModelLoader>(new SimpleWavefrontObjectModelLoader(ori)), true);

    return object_model;
}

auto RmsGaussianFilterTrackerBuilder::create_object_transition_model(
    const ObjectTransitionModelBuilder<RmsGaussianFilterTrackerBuilder::State>::
        Parameters& param) const -> StateTransition
{
    ObjectTransitionModelBuilder<State> process_builder(param);
    auto process = process_builder.build_model();

    return process;
}

std::shared_ptr<RigidBodyRenderer>
RmsGaussianFilterTrackerBuilder::create_renderer(
    const std::shared_ptr<ObjectModel>& object_model) const
{
    std::shared_ptr<RigidBodyRenderer> renderer(
        new RigidBodyRenderer(object_model->vertices(),
                              object_model->triangle_indices(),
                              camera_data_->camera_matrix(),
                              camera_data_->resolution().height,
                              camera_data_->resolution().width));

    return renderer;
}
}
