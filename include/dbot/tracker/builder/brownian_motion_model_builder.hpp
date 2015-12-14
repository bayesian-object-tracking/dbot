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
 * \file brownian_motion_model_builder.hpp
 * \date November 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include <dbot/tracker/builder/state_transition_function_builder.hpp>
#include <dbot/model/state_transition/brownian_object_motion_model.hpp>

namespace dbot
{

template <typename State, typename Input>
class BrownianMotionModelBuilder
    : public StateTransitionFunctionBuilder<State, State, Input>
{
public:
    typedef fl::StateTransitionFunction<State, State, Input> Model;
    typedef BrownianObjectMotionModel<State> DerivedModel;

    struct Parameters
    {
        double linear_acceleration_sigma;
        double angular_acceleration_sigma;
        double damping;
        double delta_time;
        int part_count;
    };

    BrownianMotionModelBuilder(const Parameters& param) : param_(param) { }

    virtual std::shared_ptr<Model> build() const
    {
        std::shared_ptr<Model> model(create());

        Eigen::MatrixXd linear_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3) *
            pow(double(param_.linear_acceleration_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3) *
            pow(double(param_.angular_acceleration_sigma), 2);

        for (size_t i = 0; i < param_.part_count; i++)
        {
            std::static_pointer_cast<DerivedModel>(model)
                ->Parameters(i,
                             Eigen::Vector3d::Zero(),
                             param_.damping,
                             linear_acceleration_covariance,
                             angular_acceleration_covariance);
        }

        return model;
    }

protected:
    std::shared_ptr<Model> create() const
    {
        std::shared_ptr<Model> model(
            new DerivedModel(param_.delta_time, param_.part_count));

        return model;
    }

private:
    Parameters param_;
};

}
