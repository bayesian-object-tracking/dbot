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

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <fl/model/process/interface/state_transition_function.hpp>

namespace dbot
{

template <typename State, typename Noise, typename Input>
class StateTransitionFunctionBuilder
{
public:
    typedef fl::StateTransitionFunction<State, Noise, Input> Model;

public:
    std::shared_ptr<Model> build() const
    {
        std::shared_ptr<Model> model(create());
        return model;
    }

protected:
    virtual std::shared_ptr<Model> create() const = 0;
};

}
