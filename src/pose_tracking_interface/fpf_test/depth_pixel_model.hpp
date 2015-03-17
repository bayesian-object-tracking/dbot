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
 * \date March 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <cstdlib>
#include <memory>

#include <Eigen/Dense>

#include <fl/distribution/gaussian.hpp>
#include <fl/model/adaptive_model.hpp>
#include <fl/model/observation/observation_model_interface.hpp>

namespace fl
{

template <typename State_> class DepthPixelModel;

template <typename State_>
struct Traits<DepthPixelModel<State_>>
{
    enum
    {
        ObsrvDim = 1,
        NoiseDim = 1,
        ParamDim = 1,
        StateDim = 1
    };

    typedef State_ State;
    typedef typename State::Scalar Scalar;

    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Obsrv;
    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;
    typedef Eigen::Matrix<Scalar, ParamDim, 1> Param;    

    typedef ObservationModelInterface<Obsrv, State, Noise> ObservationModelBase;
    typedef AdaptiveModel<Param> AdaptiveModelBase;
    typedef Gaussian<Noise> GaussianBase;

    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;
};

template <typename State>
class DepthPixelModel
    : public Traits<DepthPixelModel<State>>::ObservationModelBase,
      public Traits<DepthPixelModel<State>>::AdaptiveModelBase,
      public Traits<DepthPixelModel<State>>::GaussianBase
{
private:
    typedef DepthPixelModel This;

    typedef from_traits(Obsrv);
    typedef from_traits(Noise);
    typedef from_traits(Param);

public:
    using Traits<This>::GaussianBase::square_root;

public:
    virtual Obsrv predict_obsrv(const State& state,
                                const Noise& noise,
                                double delta_time)
    {
        Obsrv y;
        y(0) = state(0) + (sigma(param_(0)) + square_root()(0)) * noise(0);
        return y;
    }

    double sigma(double b) { return b * b * sigma_b; }

    virtual const Param& param() const { return param_; }
    virtual void param(Param params) { param_ = params; }


    virtual int obsrv_dimension() const { return Traits<This>::ObsrvDim; }
    virtual int noise_dimension() const { return Traits<This>::NoiseDim; }
    virtual int state_dimension() const { return Traits<This>::StateDim; }
    virtual int param_dimension() const { return Traits<This>::ParamDim; }

protected:
    Param param_;

public:
    double sigma_b;
};

}
