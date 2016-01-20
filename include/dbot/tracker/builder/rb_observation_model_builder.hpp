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

/**
 * \file rb_observation_model_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <ros/package.h>

#include <dbot/util/camera_data.hpp>
#include <dbot/util/object_model.hpp>
#include <dbot/util/file_shader_provider.hpp>
#include <dbot/util/default_shader_provider.hpp>
#include <dbot/util/rigid_body_renderer.hpp>
#include <dbot/model/observation/rao_blackwell_observation_model.hpp>
#include <dbot/model/observation/kinect_image_observation_model_cpu.hpp>

#ifdef DBOT_BUILD_GPU
#include <dbot/model/observation/gpu/kinect_image_observation_model_gpu.hpp>
#endif

namespace dbot
{
/**
 * \brief The NoGpuSupportException class
 */
class NoGpuSupportException : public std::exception
{
    const char* what() const noexcept
    {
        return "Tracker has not been compiled with GPU support "
               "(<PROJECT_NAME>_BUILD_GPU=OFF).";
    }
};

template <typename State>
class RbObservationModelBuilder
{
public:
    struct Parameters
    {
        /* -- Pixel occlusion process model parameters -- */
        struct Occlusion
        {
            double p_occluded_visible;
            double p_occluded_occluded;
            double initial_occlusion_prob;
        };

        /* -- Kinect pixel observation model parameters -- */
        struct Kinect
        {
            double tail_weight;
            double model_sigma;
            double sigma_factor;
        };

        /* -- Kinect image observation model parameters -- */
        bool use_gpu;
        Occlusion occlusion;
        Kinect kinect;
        double delta_time;
        int sample_count;
        bool use_custom_shaders;
        std::string vertex_shader_file;
        std::string fragment_shader_file;
        std::string geometry_shader_file;
    };

    typedef RbObservationModel<State> Model;

public:
    RbObservationModelBuilder(const std::shared_ptr<ObjectModel>& object_model,
                              const std::shared_ptr<CameraData>& camera_data,
                              const Parameters& params)
        : object_model_(object_model),
          camera_data_(camera_data),
          params_(params)
    {
    }

public:
    virtual std::shared_ptr<Model> build() const
    {
        std::shared_ptr<Model> obsrv_model;

        if (params_.use_gpu)
        {
            obsrv_model = create_gpu_based_model();
        }
        else
        {
            obsrv_model = create_cpu_based_model();
        }

        return obsrv_model;
    }

public:
    /* GPU model factor functions */
    virtual std::shared_ptr<Model> create_gpu_based_model() const
    {
#ifdef DBOT_BUILD_GPU
        auto observation_model = std::shared_ptr<Model>(
            new dbot::KinectImageObservationModelGPU<State>(
                camera_data_->camera_matrix(),
                camera_data_->resolution().height,
                camera_data_->resolution().width,
                params_.sample_count,
                object_model_->vertices(),
                object_model_->triangle_indices(),
                create_shader_provider(),
                params_.occlusion.initial_occlusion_prob,
                params_.delta_time,
                params_.occlusion.p_occluded_visible,
                params_.occlusion.p_occluded_occluded,
                params_.kinect.tail_weight,
                params_.kinect.model_sigma,
                params_.kinect.sigma_factor));

        return observation_model;
#else
        throw NoGpuSupportException();
#endif
    }

    std::shared_ptr<ShaderProvider> create_shader_provider() const
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

public:
    /* CPU model factor functions */
    virtual std::shared_ptr<Model> create_cpu_based_model() const
    {
        auto pixel_model = create_pixel_model();
        auto occlusion_process = create_occlusion_process();
        auto renderer = create_renderer();

        auto observation_model = std::shared_ptr<Model>(
            new dbot::KinectImageObservationModelCPU<fl::Real, State>(
                camera_data_->camera_matrix(),
                camera_data_->resolution().height,
                camera_data_->resolution().width,
                renderer,
                pixel_model,
                occlusion_process,
                params_.occlusion.initial_occlusion_prob,
                params_.delta_time));

        return observation_model;
    }

    virtual std::shared_ptr<KinectPixelObservationModel> create_pixel_model()
        const
    {
        std::shared_ptr<KinectPixelObservationModel>
            kinect_pixel_observation_model(
                new KinectPixelObservationModel(params_.kinect.tail_weight,
                                                params_.kinect.model_sigma,
                                                params_.kinect.sigma_factor));
        return kinect_pixel_observation_model;
    }

    virtual std::shared_ptr<OcclusionProcessModel> create_occlusion_process()
        const
    {
        std::shared_ptr<OcclusionProcessModel> occlusion_process(
            new OcclusionProcessModel(params_.occlusion.p_occluded_visible,
                                      params_.occlusion.p_occluded_occluded));

        return occlusion_process;
    }

    virtual std::shared_ptr<RigidBodyRenderer> create_renderer() const
    {
        std::shared_ptr<RigidBodyRenderer> renderer(new RigidBodyRenderer(
            object_model_->vertices(), object_model_->triangle_indices()));

        return renderer;
    }

protected:
    std::shared_ptr<ObjectModel> object_model_;
    std::shared_ptr<CameraData> camera_data_;
    Parameters params_;
};
}
