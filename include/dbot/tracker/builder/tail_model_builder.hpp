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
 * \file tail_model_builder.hpp
 * \date September 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <fl/model/observation/uniform_observation_model.hpp>
#include <fl/model/observation/linear_cauchy_observation_model.hpp>

namespace rmsgf
{

// forward declaration of TailModelFactory
template <
    typename BodyModel,
    template <typename...> class TailModelClass
>
struct TailModelFactory;

/**
 * \brief TailModelFactory for LinearCauchyObservationModel tail model
 */
template <
    typename BodyModel
>
struct TailModelFactory<BodyModel, fl::LinearCauchyObservationModel>
{
    typedef typename BodyModel::Obsrv Obsrv;
    typedef typename BodyModel::State State;

    typedef fl::LinearCauchyObservationModel<Obsrv, State> TailModel;

    template <typename Parameter>
    static TailModel create_tail_model(const Parameter& param)
    {
        auto tail = TailModel();
        auto R = tail.noise_covariance();
        tail.noise_covariance(R * std::pow(10.0 * param.obsrv_fg_noise_std, 2));

        return tail;
    }
};


/**
 * \brief TailModelFactory for UniformDepthPixelObservationModel tail model
 */
template <
    typename BodyModel
>
struct TailModelFactory<BodyModel, fl::UniformObservationModel>
{
    typedef typename BodyModel::Obsrv Obsrv;
    typedef typename BodyModel::State State;

    typedef fl::UniformObservationModel<State> TailModel;

    template <typename Parameter>
    static TailModel create_tail_model(const Parameter& param)
    {
        return TailModel(param.uniform_tail_min, param.uniform_tail_max);
    }
};

}
