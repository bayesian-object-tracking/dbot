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
 * \file rb_sensor_builder.h
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <Eigen/Dense>
#include <dbot/camera_data.h>
#include <dbot/default_shader_provider.h>
#include <dbot/file_shader_provider.h>
#include <dbot/model/kinect_image_model.h>
#include <dbot/model/kinect_pixel_model.h>
#include <dbot/model/occlusion_model.h>
#include <dbot/model/rao_blackwell_sensor.h>
#include <dbot/object_model.h>
#include <dbot/pose/euler_vector.h>
#include <dbot/rigid_body_renderer.h>
#include <memory>

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
class RbSensorBuilder
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

    typedef RbSensor<State> Model;

public:
    RbSensorBuilder(const std::shared_ptr<ObjectModel>& object_model,
                    const std::shared_ptr<CameraData>& camera_data,
                    const Parameters& params);

    virtual std::shared_ptr<Model> build() const;

public:
    /* GPU model factor functions */
    virtual std::shared_ptr<Model> create_gpu_based_model() const;

    std::shared_ptr<ShaderProvider> create_shader_provider() const;

public:
    /* CPU model factor functions */
    virtual std::shared_ptr<Model> create_cpu_based_model() const;
    virtual std::shared_ptr<KinectPixelModel> create_pixel_model() const;

    virtual std::shared_ptr<OcclusionModel> create_occlusion_process() const;

    virtual std::shared_ptr<RigidBodyRenderer> create_renderer() const;

protected:
    std::shared_ptr<ObjectModel> object_model_;
    std::shared_ptr<CameraData> camera_data_;
    Parameters params_;
};
}
