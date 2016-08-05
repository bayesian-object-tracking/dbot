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
 * J. Issac, M. Wuthrich, C. Garcia Cifuentes, J. Bohg, S. Trimpe, S. Schaal
 * Depth-Based Object Tracking Using a Robust Gaussian Filter
 * IEEE Intl Conf on Robotics and Automation, 2016
 * http://arxiv.org/abs/1602.06157
 *
 */

/**
 * \file object_transition_builder.hpp
 * \date December 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

#include <memory>

#include <fl/util/profiling.hpp>
#include <fl/util/meta.hpp>

#include <Eigen/Dense>

#include <dbot/builder/transition_function_builder.hpp>
#include <fl/model/transition/linear_transition.hpp>

namespace dbot
{
template <typename State>
struct ObjectStateTrait
{
    enum
    {
        NoiseDim = State::SizeAtCompileTime != -1 ? State::SizeAtCompileTime / 2
                                                  : Eigen::Dynamic,
        InputDim = Eigen::Dynamic
    };

    typedef Eigen::Matrix<typename State::Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<typename State::Scalar, InputDim, 1> Input;
};

template <typename State>
class ObjectTransitionBuilder
    : public TransitionFunctionBuilder<
          State,
          typename ObjectStateTrait<State>::Noise,
          typename ObjectStateTrait<State>::Input>
{
public:
    typedef fl::TransitionFunction<State,
                                        typename ObjectStateTrait<State>::Noise,
                                        typename ObjectStateTrait<State>::Input>
        Model;
    typedef fl::LinearTransition<
        State,
        typename ObjectStateTrait<State>::Noise,
        typename ObjectStateTrait<State>::Input> DerivedModel;

    struct Parameters
    {
        double linear_sigma_x;
        double linear_sigma_y;
        double linear_sigma_z;
        double angular_sigma_x;
        double angular_sigma_y;
        double angular_sigma_z;
        double velocity_factor;
        int part_count;
    };

    ObjectTransitionBuilder(const Parameters& param) : param_(param) {}
    virtual std::shared_ptr<Model> build() const
    {
        auto model = std::shared_ptr<DerivedModel>(
            new DerivedModel(build_model()));

        return std::static_pointer_cast<Model>(model);
    }

    virtual DerivedModel build_model() const
    {
        int total_state_dim = param_.part_count * 12;
        int total_noise_dim = total_state_dim / 2;

        auto model = DerivedModel(total_state_dim, total_noise_dim, 1);

        auto A = model.create_dynamics_matrix();
        auto B = model.create_noise_matrix();
        auto C = model.create_input_matrix();

        auto part_A = Eigen::Matrix<fl::Real, 12, 12>();
        auto part_B = Eigen::Matrix<fl::Real, 12, 6>();

        A.setIdentity();
        B.setZero();
        C.setZero();

        part_A.setIdentity();
        part_A.topRightCorner(6, 6).setIdentity();
        part_A.rightCols(6) *= param_.velocity_factor;

        part_B.setZero();
        part_B.block(0, 0, 3, 3) = Eigen::DiagonalMatrix<double, 3>(
                    param_.linear_sigma_x,
                    param_.linear_sigma_y,
                    param_.linear_sigma_z);
//            Eigen::Matrix3d::Identity() * param_.linear_sigma;
        part_B.block(3, 3, 3, 3) = Eigen::DiagonalMatrix<double, 3>(
                    param_.angular_sigma_x,
                    param_.angular_sigma_y,
                    param_.angular_sigma_z);

//            Eigen::Matrix3d::Identity() * param_.angular_sigma;
        part_B.bottomRows(6) = part_B.topRows(6);

        for (int i = 0; i < param_.part_count; ++i)
        {
            A.block(i * 12, i * 12, 12, 12) = part_A;
            B.block(i * 12, i * 6, 12, 6) = part_B;
        }

        model.dynamics_matrix(A);
        model.noise_matrix(B);
        model.input_matrix(C);

        return model;
    }


private:
    Parameters param_;
};
}
