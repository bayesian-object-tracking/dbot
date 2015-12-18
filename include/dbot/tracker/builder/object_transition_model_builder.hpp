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

#include <fl/util/profiling.hpp>
#include <fl/util/meta.hpp>

#include <Eigen/Dense>

#include <dbot/tracker/builder/state_transition_function_builder.hpp>
#include <fl/model/process/linear_state_transition_model.hpp>

namespace dbot
{
template <typename State>
struct ObjectStateTrait
{
    enum
    {
        NoiseDim = State::SizeAtCompileTime != -1 ? State::SizeAtCompileTime / 2
                                                  : Eigen::Dynamic
    };

    typedef Eigen::Matrix<typename State::Scalar, NoiseDim, 1> Noise;
};

template <typename State, typename Input>
class ObjectTransitionModelBuilder
    : public StateTransitionFunctionBuilder<
          State,
          typename ObjectStateTrait<State>::Noise,
          Input>
{
public:
    typedef fl::StateTransitionFunction<State,
                                        typename ObjectStateTrait<State>::Noise,
                                        Input> Model;
    typedef fl::LinearStateTransitionModel<
        State,
        typename ObjectStateTrait<State>::Noise,
        Input> DerivedModel;

    struct Parameters
    {
        double linear_sigma;
        double angular_sigma;
        double velocity_factor;
        int part_count;
    };

    ObjectTransitionModelBuilder(const Parameters& param) : param_(param) {}
    virtual std::shared_ptr<Model> build() const
    {
        auto model = std::shared_ptr<DerivedModel>(new DerivedModel(12, 6, 1));

        auto A = model->create_dynamics_matrix();
        A.setIdentity();
        A.topRightCorner(6, 6).setIdentity();
        A.rightCols(6) *= param_.velocity_factor;
        model->dynamics_matrix(A);

        Eigen::MatrixXd B = model->create_noise_matrix();
        B.setZero();
        B.block(6, 0, 3, 3) = Eigen::Matrix3d::Identity() * param_.linear_sigma;
        B.block(9, 3, 3, 3) =
            Eigen::Matrix3d::Identity() * param_.angular_sigma;
        B.topRows(6) = B.bottomRows(6);
        model->noise_matrix(B);

        auto C = model->create_input_matrix();
        C.setZero();
        model->input_matrix(C);

        return std::static_pointer_cast<Model>(model);
    }

protected:
    std::shared_ptr<Model> create() const
    {
        std::shared_ptr<Model> model(new DerivedModel(12, 6, 1));

        return model;
    }

private:
    Parameters param_;
};
}
