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

/**
 * \file rb_sensor_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <dbot/builder/rb_sensor_builder.h>
#include <dbot/model/kinect_image_model.h>

#ifdef DBOT_BUILD_GPU
#include <dbot/gpu/kinect_image_model_gpu.h>
#endif

namespace dbot
{
template <typename State>
RbSensorBuilder<State>::RbSensorBuilder(
    const std::shared_ptr<ObjectModel>& object_model,
    const std::shared_ptr<CameraData>& camera_data,
    const Parameters& params)
    : object_model_(object_model), camera_data_(camera_data), params_(params)
{
}

template <typename State>
auto RbSensorBuilder<State>::build() const -> std::shared_ptr<Model>
{
    std::shared_ptr<Model> sensor;

    if (params_.use_gpu)
    {
        sensor = create_gpu_based_model();
    }
    else
    {
        sensor = create_cpu_based_model();
    }

    return sensor;
}

template <typename State>
auto RbSensorBuilder<State>::create_gpu_based_model() const
    -> std::shared_ptr<Model>
{
#ifdef DBOT_BUILD_GPU
    auto sensor = std::shared_ptr<Model>(new dbot::KinectImageModelGPU<State>(
        camera_data_->camera_matrix(),
        camera_data_->resolution().height,
        camera_data_->resolution().width,
        params_.sample_count,
        object_model_->vertices(),
        object_model_->triangle_indices(),
        create_shader_provider(),
        false,  // TODO should be a parameter from the config file
        false,  // TODO should be a parameter from the config file
        params_.occlusion.initial_occlusion_prob,
        params_.delta_time,
        params_.occlusion.p_occluded_visible,
        params_.occlusion.p_occluded_occluded,
        params_.kinect.tail_weight,
        params_.kinect.model_sigma,
        params_.kinect.sigma_factor));

    return sensor;
#else
    throw NoGpuSupportException();
#endif
}

template <typename State>
auto RbSensorBuilder<State>::create_shader_provider() const
    -> std::shared_ptr<ShaderProvider>
{
    if (params_.use_custom_shaders)
    {
        return std::shared_ptr<ShaderProvider>(
            new FileShaderProvider(params_.fragment_shader_file,
                                   params_.vertex_shader_file,
                                   params_.geometry_shader_file));
    }

    return std::shared_ptr<ShaderProvider>(new DefaultShaderProvider());
}

template <typename State>
auto RbSensorBuilder<State>::create_cpu_based_model() const
    -> std::shared_ptr<Model>
{
    auto pixel_model = create_pixel_model();
    auto occlusion_process = create_occlusion_process();
    auto renderer = create_renderer();

    auto sensor =
        std::shared_ptr<Model>(new dbot::KinectImageModel<fl::Real, State>(
            camera_data_->camera_matrix(),
            camera_data_->resolution().height,
            camera_data_->resolution().width,
            renderer,
            pixel_model,
            occlusion_process,
            params_.occlusion.initial_occlusion_prob,
            params_.delta_time));

    return sensor;
}

template <typename State>
auto RbSensorBuilder<State>::create_pixel_model() const
    -> std::shared_ptr<KinectPixelModel>
{
    std::shared_ptr<KinectPixelModel> kinect_pixel_sensor(
        new KinectPixelModel(params_.kinect.tail_weight,
                             params_.kinect.model_sigma,
                             params_.kinect.sigma_factor));
    return kinect_pixel_sensor;
}

template <typename State>
auto RbSensorBuilder<State>::create_occlusion_process() const
    -> std::shared_ptr<OcclusionModel>
{
    std::shared_ptr<OcclusionModel> occlusion_process(
        new OcclusionModel(params_.occlusion.p_occluded_visible,
                           params_.occlusion.p_occluded_occluded));

    return occlusion_process;
}

template <typename State>
auto RbSensorBuilder<State>::create_renderer() const
    -> std::shared_ptr<RigidBodyRenderer>
{
    std::shared_ptr<RigidBodyRenderer> renderer(new RigidBodyRenderer(
        object_model_->vertices(), object_model_->triangle_indices()));

    return renderer;
}
}
