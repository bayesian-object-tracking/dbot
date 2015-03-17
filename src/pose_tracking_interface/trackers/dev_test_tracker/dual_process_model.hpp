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
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <Eigen/Dense>

#include <fl/util/meta.hpp>

#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/process/joint_process_model.hpp>
#include <pose_tracking/models/process_models/brownian_object_motion_model.hpp>

namespace fl
{

template <
    typename PoseProcessModel,
    typename ParametersProcessModel>
class DualProcessModel;

template <
    typename PoseProcessModel,
    typename ParametersProcessModel>
struct Traits<
           DualProcessModel<PoseProcessModel, ParametersProcessModel>>
{    
    typedef typename Traits<PoseProcessModel>::State ObjectState;
    typedef typename Traits<ParametersProcessModel>::State ParametersState;
    typedef typename ObjectState::Scalar Scalar;

    typedef Eigen::Matrix<
                Scalar,
                JoinSizes<
                    ObjectState::RowsAtCompileTime,
                    ParametersState::RowsAtCompileTime
                >::Size,
                1
            > State;

    typedef Eigen::Matrix<
                Scalar,
                JoinSizes<
                    ObjectState::RowsAtCompileTime,
                    ParametersState::RowsAtCompileTime
                >::Size,
                1
            > Noise;

    typedef Eigen::Matrix<
                Scalar,
                JoinSizes<
                    ObjectState::RowsAtCompileTime,
                    ParametersState::RowsAtCompileTime
                >::Size,
                1
            > Input;

    typedef ProcessModelInterface<State, Noise, Input> ProcessModelBase;
};

template <
    typename PoseProcessModel,
    typename ParametersProcessModel>
class DualProcessModel
    : public Traits<
                 DualProcessModel<PoseProcessModel, ParametersProcessModel>
             >::ProcessModelBase
{
public:
    typedef DualProcessModel<PoseProcessModel, ParametersProcessModel> This;

    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::Input Input;
    typedef typename Traits<This>::Scalar Scalar;

public:
    DualProcessModel(
        const std::shared_ptr<PoseProcessModel>& pose_process_model,
        const std::shared_ptr<ParametersProcessModel>& parameter_process_model)
        : pose_process_model_(pose_process_model),
          parameters_process_model_(parameter_process_model)
    {
    }

    ~DualProcessModel() { }

    virtual State predict_state(double delta_time,
                                const State& state,
                                const Noise& noise,
                                const Input& input)
    {
        State prediction(state_dimension(), 1);

        prediction.topRows(pose_process_model_->state_dimension()) =
            pose_process_model_->predict_state(
                delta_time,
                state.topRows(pose_process_model_->state_dimension()),
                noise.topRows(pose_process_model_->noise_dimension()),
                input.topRows(pose_process_model_->input_dimension()));

        prediction.bottomRows(parameters_process_model_->state_dimension()) =
            parameters_process_model_->predict_state(
                delta_time,
                state.bottomRows(parameters_process_model_->state_dimension()),
                noise.bottomRows(parameters_process_model_->noise_dimension()),
                input.bottomRows(parameters_process_model_->input_dimension()));

        return prediction;
    }

    virtual size_t state_dimension() const
    {
        return pose_process_model_->state_dimension()
               + parameters_process_model_->state_dimension();
    }

    virtual size_t noise_dimension() const
    {
        return pose_process_model_->noise_dimension()
               + parameters_process_model_->noise_dimension();
    }

    virtual size_t input_dimension() const
    {
        return pose_process_model_->input_dimension()
               + parameters_process_model_->input_dimension();
    }

    const std::shared_ptr<PoseProcessModel>& pose_process_model()
    {
        return pose_process_model_;
    }

    const std::shared_ptr<ParametersProcessModel>& parameters_process_model()
    {
        return parameters_process_model_;
    }

protected:
    std::shared_ptr<PoseProcessModel> pose_process_model_;
    std::shared_ptr<ParametersProcessModel> parameters_process_model_;
};

}
