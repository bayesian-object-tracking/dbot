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
 * \date 05/25/2014
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * \author Jan Issac (jan.issac@gmail.com)
 */

#pragma once

//TODO: THIS IS A LINEAR GAUSSIAN PROCESS, THIS CLASS SHOULD DISAPPEAR

#include <Eigen/Dense>

#include <fl/util/assertions.hpp>
//#include <dbot/model/state_transition/stationary_process_model.hpp>
//#include <fast_filtering/distributions/gaussian.hpp>
#include <fl/distribution/gaussian.hpp>

namespace dbot
{

// Forward declarations
template <typename State> class DampedWienerProcessModel;

namespace internal
{
/**
 * DampedWienerProcess distribution traits specialization
 * \internal
 */
template <typename State_>
struct Traits<DampedWienerProcessModel<State_> >
{
    typedef State_                                              State;
    typedef typename State::Scalar                              Scalar;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Input;
    typedef Eigen::Matrix<Scalar, State::SizeAtCompileTime, 1>  Noise;

    typedef fl::Gaussian<Noise>                 GaussianType;
    typedef typename GaussianType::SecondMoment SecondMoment;

//    typedef StationaryProcessModel<State, Input>   ProcessModelBase;
//    typedef GaussianMap<State, Noise>         GaussianMapBase;
};
}

/**
 * \class DampedWienerProcess
 *
 * \ingroup distributions
 * \ingroup process_models
 */
template <typename State_>
class DampedWienerProcessModel
//        :public internal::Traits<DampedWienerProcessModel<State> >::ProcessModelBase
//        ,public internal::Traits<DampedWienerProcessModel<State> >::GaussianMapBase
{
public:
    typedef internal::Traits<DampedWienerProcessModel<State_> > Traits;

    typedef typename Traits::Scalar         Scalar;
    typedef typename Traits::State         State;

    typedef typename Traits::SecondMoment       SecondMoment;
    typedef typename Traits::Input          Input;
    typedef typename Traits::Noise          Noise;
    typedef typename Traits::GaussianType   GaussianType;

public:

    /// \todo uncomment default argument
    explicit DampedWienerProcessModel(
            const double& delta_time,
            const unsigned& dimension = State::SizeAtCompileTime):

        gaussian_(dimension),
        delta_time_(delta_time)

    {
        // check that state is derived from eigen
        static_assert_base(State, Eigen::Matrix<typename State::Scalar, State::SizeAtCompileTime, 1>);
    }

    virtual ~DampedWienerProcessModel() noexcept { }

    virtual State MapStandardGaussian(const Noise& sample) const
    {
        return gaussian_.map_standard_normal(sample);
    }

    virtual void Condition(const State&  state,
                           const Input&   input)
    {
        gaussian_.mean(Mean(delta_time_, state, input));
        gaussian_.diagonal_covariance(Covariance(delta_time_));
    }

    virtual State state(const State& prev_state,
                        const Noise& noise,
                        const Input& input)
    {
        Condition(prev_state, input);
        return MapStandardGaussian(noise);

    }


//    virtual void Condition(const Scalar&  delta_time,
//                           const State&  state,
//                           const Input&   input)
//    {
//        gaussian_.mean(Mean(delta_time, state, input));
//        gaussian_.diagonal_covariance(Covariance(delta_time));
//    }


    virtual void Parameters(const Scalar& damping,
                            const SecondMoment& noise_covariance)
    {
        damping_ = damping;
        noise_covariance_ = noise_covariance;
    }

    virtual unsigned Dimension() const
    {
        return noise_dimension(); // all dimensions are the same
    }


    virtual int noise_dimension() const
    {
        return gaussian_.dimension();
    }


private:
    State Mean(const Scalar& delta_time,
                    const State& state,
                    const Input& input)
    {
        if(damping_ == 0)
            return state + delta_time * input;

        State state_expectation = (1.0 - exp(-damping_*delta_time)) / damping_ * input +
                                              exp(-damping_*delta_time)  * state;

        // if the damping_ is too small, the result might be nan, we thus return the limit for damping_ -> 0
        if(!std::isfinite(state_expectation.norm()))
            state_expectation = state + delta_time * input;

        return state_expectation;
    }

    SecondMoment Covariance(const Scalar& delta_time)
    {
        if(damping_ == 0)
            return delta_time * noise_covariance_;

        Scalar factor = (1.0 - exp(-2.0*damping_*delta_time))/(2.0*damping_);
        if(!std::isfinite(factor))
            factor = delta_time;

        return factor * noise_covariance_;
    }

private:
    double delta_time_;


    // conditional
    GaussianType gaussian_;

    // parameters
    Scalar damping_;
    SecondMoment noise_covariance_;

    // euler-mascheroni constant
    static constexpr Scalar gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

