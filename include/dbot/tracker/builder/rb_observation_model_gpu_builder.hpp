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
        std::string vertex_shader_path =
            ros::package::getPath("dbot") + "/src/dbot/model/observation/" +
            "gpu/shaders/" + "VertexShader.vertexshader";

        std::string fragment_shader_path =
            ros::package::getPath("dbot") + "/src/dbot/model/observation/" +
            "gpu/shaders/" + "FragmentShader.fragmentshader";

        dbot::FileShaderProvider shader_provider(fragment_shader_path,
                                                 vertex_shader_path);

        std::shared_ptr<Model> observation_model(
            new Model(camera_data_.camera_matrix(),
                      camera_data_.resolution().height,
                      camera_data_.resolution().width,
                      param_.max_sample_count,
                      object_model_.vertices(),
                      object_model_.triangle_indices(),
                      shader_provider,
                      param_.initial_occlusion_prob,
                      param_.delta_time,
                      param_.p_occluded_visible,
                      param_.p_occluded_occluded,
                      param_.tail_weight,
                      param_.model_sigma,
                      param_.sigma_factor));

        return observation_model;
    }

private:
    Parameters param_;
    ObjectModel object_model_;
    CameraData camera_data_;
};
}
