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
 * \file rb_observation_model_gpu_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <ros/package.h>

#include <dbot/util/object_model.hpp>
#include <dbot/util/camera_data.hpp>
#include <dbot/util/file_shader_provider.hpp>
#include <dbot/util/default_shader_provider.hpp>
#include <dbot/tracker/builder/rb_observation_model_builder.hpp>
#include <dbot/model/observation/gpu/kinect_image_observation_model_gpu.hpp>

namespace dbot
{
template <typename State>
class RbObservationModelGpuBuilder : public RbObservationModelBuilder<State>
{
public:
    typedef RbObservationModel<State> BaseModel;
    typedef RbObservationModelBuilder<State> Base;
    typedef KinectImageObservationModelGPU<State> Model;

    typedef typename Model::Observation Obsrv;
    typedef typename Base::Parameters Parameters;

public:
    RbObservationModelGpuBuilder(const Parameters& param,
                                 const ObjectModel& object_model,
                                 const CameraData& camera_data)
        : param_(param), object_model_(object_model), camera_data_(camera_data)
    {
    }

protected:
    virtual std::shared_ptr<BaseModel> create() const
    {
        std::shared_ptr<Model> observation_model(
            new Model(camera_data_.camera_matrix(),
                      camera_data_.resolution().height,
                      camera_data_.resolution().width,
                      param_.sample_count,
                      object_model_.vertices(),
                      object_model_.triangle_indices(),
                      create_shader_provider(),
                      param_.occlusion.initial_occlusion_prob,
                      param_.delta_time,
                      param_.occlusion.p_occluded_visible,
                      param_.occlusion.p_occluded_occluded,
                      param_.kinect.tail_weight,
                      param_.kinect.model_sigma,
                      param_.kinect.sigma_factor));

        return observation_model;
    }

    std::shared_ptr<ShaderProvider> create_shader_provider() const
    {
        if (param_.use_custom_shaders)
        {
            return std::shared_ptr<ShaderProvider>(
                new FileShaderProvider(param_.fragment_shader_file,
                                       param_.vertex_shader_file,
                                       param_.geometry_shader_file));
        }

        return std::shared_ptr<ShaderProvider>(new DefaultShaderProvider());
    }

private:
    Parameters param_;
    ObjectModel object_model_;
    CameraData camera_data_;
};
}
