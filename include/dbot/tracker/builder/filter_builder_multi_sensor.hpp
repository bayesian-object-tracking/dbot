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
 * \file filter_builder_hpp.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include "filter_builder_base.hpp"

#include <dbot/util/rigid_body_renderer.hpp>
#include <dbot/model/observation/depth_pixel_observation_model.hpp>

#include <fl/model/observation/joint_observation_model_iid.hpp>
#include <fl/filter/gaussian/multi_sensor_gaussian_filter.hpp>

namespace rmsgf
{

/**
 * \brief FilterBuilder for MultiSensorGaussianFilter
 */
template <
    typename Quadrature,
    template <typename...> class TailModel,
    typename StateType
>
struct FilterBuilder<
           fl::MultiSensorGaussianFilter,
           TailModel,
           Quadrature,
           StateType>
    : FilterBuilderBase<fl::MultiSensorGaussianFilter, Quadrature, StateType>
{
    typedef FilterBuilderBase<
                fl::MultiSensorGaussianFilter, Quadrature, StateType
            > Base;

    typedef typename Base::State State;
    typedef typename Base::Parameter Parameter;
    typedef typename Base::LinearStateModel LinearStateModel;
    using Base::create_state_transtion_model;

    /* ---------------------------------------------------------------------- */
    /* - Observation Model                                                  - */
    /* ---------------------------------------------------------------------- */
    // Pixel Level
    typedef fl::DepthPixelObservationModel<State> PixelModel;

    // Image Level
    typedef fl::JointObservationModel<
                fl::MultipleOf<PixelModel, Eigen::Dynamic>
            > ObsrvModel;

    /* ---------------------------------------------------------------------- */
    /* - Filter                                                             - */
    /* ---------------------------------------------------------------------- */
    typedef fl::MultiSensorGaussianFilter<
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
        auto quadrature = Quadrature();

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

        return ObsrvModel(pixel_obsrv_model, param.sensors);
    }

    template <typename Filter, typename Pose>
    static void set_nominal_pose(Filter& filter, Pose& pose)
    {
        filter.obsrv_model().local_obsrv_model().nominal_pose(pose);
    }
};

template <
    typename Quadrature,
    template <typename...> class TailModel,
    typename StateType
>
struct FilterBuilder<
           fl::GaussianFilter,
           TailModel,
           Quadrature,
           StateType>
    : FilterBuilderBase<fl::GaussianFilter, Quadrature, StateType>
{
    typedef FilterBuilderBase<
                fl::GaussianFilter, Quadrature, StateType
            > Base;

    typedef typename Base::State State;
    typedef typename Base::Parameter Parameter;
    typedef typename Base::LinearStateModel LinearStateModel;
    using Base::create_state_transtion_model;

    /* ---------------------------------------------------------------------- */
    /* - Observation Model                                                  - */
    /* ---------------------------------------------------------------------- */
    // Pixel Level
    typedef fl::DepthPixelObservationModel<State> PixelModel;

    // Image Level
    typedef fl::JointObservationModel<
                fl::MultipleOf<PixelModel, Eigen::Dynamic>
            > ObsrvModel;

    /* ---------------------------------------------------------------------- */
    /* - Filter                                                             - */
    /* ---------------------------------------------------------------------- */
    typedef fl::GaussianFilter<
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
        auto quadrature = Quadrature();

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

        return ObsrvModel(pixel_obsrv_model, param.sensors);
    }

    template <typename Filter, typename Pose>
    static void set_nominal_pose(Filter& filter, Pose& pose)
    {
        filter.obsrv_model().local_obsrv_model().nominal_pose(pose);
    }
};

}
