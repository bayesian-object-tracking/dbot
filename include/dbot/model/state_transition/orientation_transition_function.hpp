/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \file orientation_state_transition_model.hpp
 * \date July 2015
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>

#include <fl/util/traits.hpp>
#include <osr/euler_vector.hpp>

#include <fl/model/process/interface/state_transition_function.hpp>
#include <fl/model/process/linear_state_transition_model.hpp>

namespace osr
{


class OSTFTypes
{
public:
    typedef Eigen::Matrix<fl::Real, 6,1> State;
    typedef Eigen::Matrix<fl::Real, 3,1> Noise;
    typedef Eigen::Matrix<fl::Real, 3,1> Input;

    typedef Eigen::Matrix<fl::Real, 3,1> Orientation;
    typedef Eigen::Matrix<fl::Real, 3,1> Delta;
};



class OrientationStateTransitionFunction:
       public fl::StateTransitionFunction<OSTFTypes::State,
                                      OSTFTypes::Noise,
                                      OSTFTypes::Input>
{
public:
    typedef OSTFTypes::State            State;
    typedef OSTFTypes::Noise            Noise;
    typedef OSTFTypes::Input            Input;
    typedef OSTFTypes::Orientation      Orientation;
    typedef OSTFTypes::Delta  Delta;

    typedef fl::LinearStateTransitionModel<Delta, Input> DeltaModel;

    typedef typename DeltaModel::DynamicsMatrix DynamicsMatrix;
    typedef typename DeltaModel::DynamicsMatrix NoiseMatrix;
    typedef typename DeltaModel::InputMatrix    InputMatrix;

public:
    OrientationStateTransitionFunction() { }

    virtual ~OrientationStateTransitionFunction() noexcept { }

public:
    // state format: (euler_vector, angular_velocity)
    virtual State state(const State& prev_state,
                        const Noise& noise,
                        const Input& input) const
    {
        EulerVector prev_orientation = prev_state.topRows(3);
        Delta prev_delta = prev_state.bottomRows(3);

        EulerVector orientation = EulerVector(prev_delta) * prev_orientation;
        Delta delta = delta_model_.state(prev_delta, noise, input);

        State next_state; next_state << orientation, delta;
        return next_state;
    }




    /// factory functions ******************************************************
    virtual InputMatrix create_input_matrix() const
    {
        return delta_model_.create_input_matrix();
    }

    virtual DynamicsMatrix create_dynamics_matrix() const
    {
        return delta_model_.create_dynamics_matrix();
    }

    virtual NoiseMatrix create_noise_matrix() const
    {
        return delta_model_.create_noise_matrix();
    }

    /// accessors **************************************************************
    virtual const DynamicsMatrix& dynamics_matrix() const
    {
        return delta_model_.dynamics_matrix();
    }

    virtual const InputMatrix& input_matrix() const
    {
        return delta_model_.input_matrix();
    }

    virtual const NoiseMatrix& noise_matrix() const
    {
        return delta_model_.noise_matrix();
    }

    virtual const NoiseMatrix& noise_covariance() const
    {
        return delta_model_.noise_covariance();
    }

    virtual int state_dimension() const
    {
        return fl::DimensionOf<State>::Value;
    }

    virtual int noise_dimension() const
    {
        return fl::DimensionOf<Noise>::Value;
    }

    virtual int input_dimension() const
    {
        return fl::DimensionOf<Input>::Value;
    }

    /// mutators ***************************************************************
    virtual void dynamics_matrix(const DynamicsMatrix& dynamics_mat)
    {
        delta_model_.dynamics_matrix(dynamics_mat);
    }

    virtual void input_matrix(const InputMatrix& input_mat)
    {
        delta_model_.input_matrix(input_mat);
    }

    virtual void noise_matrix(const NoiseMatrix& noise_mat)
    {
        delta_model_.noise_matrix(noise_mat);
    }

    virtual void noise_covariance(const NoiseMatrix& noise_mat_squared)
    {
        delta_model_.noise_covariance(noise_mat_squared);
    }


protected:
    DeltaModel delta_model_;
};

}


