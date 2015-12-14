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
 * \file filter_builder.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <dbot/util/rigid_body_renderer.hpp>

#include <dbot/model/observation/depth_pixel_observation_model.hpp>
#include <fl/model/observation/linear_gaussian_observation_model.hpp>
#include <fl/model/observation/body_tail_observation_model.hpp>
#include <fl/filter/gaussian/robust_multi_sensor_gaussian_filter.hpp>

#include "filter_builder_base.hpp"
#include "tail_model_builder.hpp"

namespace rmsgf
{

/**
 * \brief FilterBuilder for RobustMultiSensorGaussianFilter
 */
template <
    template <typename...> class TailModelClass,
    typename Quadrature,
    typename StateType
>
struct FilterBuilder<
           fl::RobustMultiSensorGaussianFilter,
           TailModelClass,
           Quadrature,
           StateType>
    : FilterBuilderBase<
            fl::RobustMultiSensorGaussianFilter, Quadrature, StateType>
{
    typedef FilterBuilderBase<
                fl::RobustMultiSensorGaussianFilter, Quadrature, StateType
            > Base;

    typedef typename Base::State State;
    typedef typename Base::Parameter Parameter;
    typedef typename Base::LinearStateModel LinearStateModel;
    using Base::create_state_transtion_model;

    /* ---------------------------------------------------------------------- */
    /* - Observation Model                                                  - */
    /* ---------------------------------------------------------------------- */
    // Pixel Level: Body model
    typedef fl::DepthPixelObservationModel<State> PixelModel;

    // Pixel Level: Tail model
    typedef TailModelFactory<PixelModel, TailModelClass> TailFactory;
    typedef typename TailFactory::TailModel TailModel;

    // Pixel Level: Body-Tail model
    typedef fl::BodyTailObsrvModel<
                PixelModel, TailModel
            > BodyTailPixelModel;

    // Image Level: create many of BodyTailPixelModel
    typedef fl::JointObservationModel<
                fl::MultipleOf<BodyTailPixelModel, Eigen::Dynamic>
            > ObsrvModel;

    /* ---------------------------------------------------------------------- */
    /* - Filter                                                             - */
    /* ---------------------------------------------------------------------- */
    typedef fl::RobustMultiSensorGaussianFilter<
                LinearStateModel,
                ObsrvModel,
                Quadrature
            > Filter;

    typedef typename fl::Traits<Filter>::Belief Belief;
    typedef typename fl::Traits<Filter>::Obsrv Obsrv;

    /**
     * \brief Builds the Robust multi-sensor Gaussian filter
     */
    Filter build_filter(
        const std::shared_ptr<dbot::RigidBodyRenderer>& renderer,
        const Parameter& param)
    {
        /* ------------------------------ */
        /* - State transition model     - */
        /* ------------------------------ */
        auto state_transition_model =
            create_state_transtion_model(param, State());

        /* ------------------------------ */
        /* - Observation model          - */
        /* ------------------------------ */
        auto obsrv_model = create_obsrv_model(renderer, param);

        /* ------------------------------ */
        /* - Quadrature                 - */
        /* ------------------------------ */
        auto quadrature = Quadrature(param.ut_alpha);

        /* ------------------------------ */
        /* - Filter                     - */
        /* ------------------------------ */
        return Filter(state_transition_model, obsrv_model, quadrature);
    }

    /**
     * \brief Constructs a joint observation model which is a joint  model
     *   of multiple heavy tail pixel observation models (Body-Tail models
     *   with a depth pixel body model and a linear Cauchy tail model.
     */
    ObsrvModel create_obsrv_model(
        const std::shared_ptr<dbot::RigidBodyRenderer>& renderer,
        const Parameter& param)
    {
        auto pixel_obsrv_model = PixelModel(
            renderer,
            param.obsrv_bg_depth,
            param.obsrv_fg_noise_std,
            param.obsrv_bg_noise_std);

        auto body_tail_pixel_model =
            BodyTailPixelModel(
                pixel_obsrv_model,
                TailFactory::create_tail_model(param),
                param.obsrv_body_tail_weight);

        return ObsrvModel(body_tail_pixel_model, param.sensors);
    }

    template <typename Filter, typename Pose>
    static void set_nominal_pose(Filter& filter, Pose& pose)
    {
        filter.obsrv_model().local_obsrv_model().body_model().nominal_pose(pose);
    }
};

}
