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

#include <cstdlib>
#include <memory>

#include <Eigen/Dense>

#include <fl/model/observation/factorized_iid_observation_model.hpp>

namespace fl
{

template <typename Scalar> class PixelObservationModel;

template <typename Scalar_>
struct Traits<
           PixelObservationModel<Scalar_>>
{
    enum
    {
        ObsrvDim = 2,
        NoiseDim = 1,
        StateDim = 2
    };

    typedef Scalar_ Scalar;

    // [y  y^2]
    typedef Eigen::Matrix<Scalar, ObsrvDim, 1> Observation;

    typedef Eigen::Matrix<Scalar, NoiseDim, 1> Noise;

    // [h_i(x) h_i(x)^2] rendered pixel
    typedef Eigen::Matrix<Scalar, StateDim, 1> State;

    typedef Gaussian<Noise> GaussianBase;
    typedef typename Traits<GaussianBase>::SecondMoment SecondMoment;

    typedef ObservationModelInterface<
                Observation,
                State,
                Noise
            > ObservationModelBase;
};

template <typename Scalar>
class PixelObservationModel
    : public Traits<PixelObservationModel<Scalar>>::GaussianBase,
      public Traits<PixelObservationModel<Scalar>>::ObservationModelBase
{
public:
    typedef PixelObservationModel<Scalar> This;

    typedef typename Traits<This>::Observation Observation;
    typedef typename Traits<This>::State State;
    typedef typename Traits<This>::Noise Noise;
    typedef typename Traits<This>::SecondMoment SecondMoment;

    using Traits<This>::GaussianBase::mean;
    using Traits<This>::GaussianBase::covariance;
    using Traits<This>::GaussianBase::dimension;

public:
    PixelObservationModel(double variance)
    {
        covariance(SecondMoment::Identity() * variance);
    }

    virtual Observation predict_observation(const State& state,
                                            const Noise& noise,
                                            double delta_time)
    {
        Observation y;

        y(0) = state(0)
               + std::exp(state(1)) * std::sqrt(covariance()(0,0)) * noise(0);
        y(1) = y(0) * y(0);

        return y;
    }

    virtual size_t observation_dimension() const
    {
        return Traits<This>::ObsrvDim;
    }

    virtual size_t noise_dimension() const
    {
        return Traits<This>::NoiseDim;
    }

    virtual size_t state_dimension() const
    {
        return Traits<This>::StateDim;
    }
};

}
