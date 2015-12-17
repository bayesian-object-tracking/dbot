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
 * \file rb_observation_model_cpu_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <dbot/util/object_model.hpp>
#include <dbot/util/camera_data.hpp>
#include <dbot/tracker/builder/rb_observation_model_builder.hpp>
#include <dbot/model/observation/kinect_image_observation_model_cpu.hpp>

namespace dbot
{
template <typename State>
class RbObservationModelCpuBuilder : public RbObservationModelBuilder<State>
{
public:
    typedef KinectImageObservationModelCPU<fl::Real, State> Model;
    typedef typename Model::Observation Obsrv;
    typedef RbObservationModel<State> BaseModel;

    typedef RbObservationModelBuilder<State> Base;
    typedef typename Base::Parameters Parameters;

public:
    RbObservationModelCpuBuilder(const Parameters& param,
                                 const ObjectModel& object_model,
                                 const CameraData& camera_data)
        : param_(param), object_model_(object_model), camera_data_(camera_data)
    {
    }

protected:
    virtual std::shared_ptr<BaseModel> create() const
    {
        auto pixel_model = create_pixel_model();
        auto occlusion_process = create_occlusion_process();
        auto renderer = create_renderer();

        auto observation_model = std::shared_ptr<Model>(
            new Model(camera_data_.camera_matrix(),
                      camera_data_.resolution().height,
                      camera_data_.resolution().width,
                      renderer,
                      pixel_model,
                      occlusion_process,
                      param_.occlusion.initial_occlusion_prob,
                      param_.delta_time));

        return observation_model;
    }

protected:
    virtual std::shared_ptr<KinectPixelObservationModel> create_pixel_model()
        const
    {
        std::shared_ptr<KinectPixelObservationModel>
            kinect_pixel_observation_model(
                new KinectPixelObservationModel(param_.kinect.tail_weight,
                                                param_.kinect.model_sigma,
                                                param_.kinect.sigma_factor));
        return kinect_pixel_observation_model;
    }

    virtual std::shared_ptr<OcclusionProcessModel> create_occlusion_process()
        const
    {
        std::shared_ptr<OcclusionProcessModel> occlusion_process(
            new OcclusionProcessModel(param_.occlusion.p_occluded_visible,
                                      param_.occlusion.p_occluded_occluded));

        return occlusion_process;
    }

    virtual std::shared_ptr<RigidBodyRenderer> create_renderer() const
    {
        std::shared_ptr<RigidBodyRenderer> renderer(new RigidBodyRenderer(
            object_model_.vertices(), object_model_.triangle_indices()));

        return renderer;
    }

private:
    Parameters param_;
    ObjectModel object_model_;
    CameraData camera_data_;
};
}
