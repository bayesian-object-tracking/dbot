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

#include <dbot/builder/gaussian_tracker_builder.h>
#include <dbot/simple_wavefront_object_loader.h>

namespace dbot
{
GaussianTrackerBuilder::GaussianTrackerBuilder(
    const Parameters& param,
    const std::shared_ptr<CameraData>& camera_data)
    : param_(param), camera_data_(camera_data)
{
}

std::shared_ptr<GaussianTracker> GaussianTrackerBuilder::build()
{
    auto object_model = create_object_model(param_.ori);

    auto filter = create_filter(object_model);

    auto tracker =
        std::make_shared<GaussianTracker>(filter,
                                          object_model,
                                          param_.moving_average_update_rate,
                                          param_.center_object_frame);

    return tracker;
}

auto GaussianTrackerBuilder::create_filter(
    const std::shared_ptr<ObjectModel>& object_model) -> std::shared_ptr<Filter>
{
    /* ------------------------------ */
    /* - State transition model     - */
    /* ------------------------------ */
    auto transition = create_object_transition(param_.object_transition);

    /* ------------------------------ */
    /* - Observation model          - */
    /* ------------------------------ */
    auto sensor = create_sensor(object_model, camera_data_, param_.observation);

    /* ------------------------------ */
    /* - Quadrature                 - */
    /* ------------------------------ */
    auto quadrature = Quadrature(param_.ut_alpha);

    /* ------------------------------ */
    /* - Filter                     - */
    /* ------------------------------ */
    auto filter =
        std::shared_ptr<Filter>(new Filter(transition, sensor, quadrature));

    return filter;
}

auto GaussianTrackerBuilder::create_sensor(
    const std::shared_ptr<ObjectModel>& object_model,
    const std::shared_ptr<CameraData>& camera_data,
    const Parameters::Observation& param) const -> Sensor
{
    typedef GaussianTracker::PixelModel PixelModel;
    typedef GaussianTracker::TailModel TailModel;
    typedef GaussianTracker::BodyTailPixelModel BodyTailModel;

    auto renderer = create_renderer(object_model);

    auto pixel_sensor = PixelModel(
        renderer, param.bg_depth, param.fg_noise_std, param.bg_noise_std);

    auto tail_sensor =
        TailModel(param.uniform_tail_min, param.uniform_tail_max);

    auto body_tail_pixel_model =
        BodyTailModel(pixel_sensor, tail_sensor, param.tail_weight);

    return Sensor(body_tail_pixel_model, camera_data->pixels());
}

std::shared_ptr<ObjectModel> GaussianTrackerBuilder::create_object_model(
    const ObjectResourceIdentifier& ori) const
{
    auto object_model = std::make_shared<ObjectModel>(
        std::shared_ptr<ObjectModelLoader>(
            new SimpleWavefrontObjectModelLoader(ori)),
        param_.center_object_frame);

    return object_model;
}

auto GaussianTrackerBuilder::create_object_transition(
    const ObjectTransitionBuilder<GaussianTrackerBuilder::State>::Parameters&
        param) const -> Transition
{
    ObjectTransitionBuilder<State> process_builder(param);
    auto process = process_builder.build_model();

    return process;
}

std::shared_ptr<RigidBodyRenderer> GaussianTrackerBuilder::create_renderer(
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
