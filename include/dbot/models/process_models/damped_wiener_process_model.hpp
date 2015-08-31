/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 05/25/2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_DAMPED_WIENER_PROCESS_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_DAMPED_WIENER_PROCESS_MODEL_HPP

//TODO: THIS IS A LINEAR GAUSSIAN PROCESS, THIS CLASS SHOULD DISAPPEAR

#include <Eigen/Dense>

#include <fl/util/assertions.hpp>
//#include <dbot/models/process_models/stationary_process_model.hpp>
//#include <fast_filtering/distributions/gaussian.hpp>
#include <fl/distribution/gaussian.hpp>

namespace ff
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
template <typename State>
class DampedWienerProcessModel
//        :public internal::Traits<DampedWienerProcessModel<State> >::ProcessModelBase
//        ,public internal::Traits<DampedWienerProcessModel<State> >::GaussianMapBase
{
public:
    typedef internal::Traits<DampedWienerProcessModel<State> > Traits;

    typedef typename Traits::Scalar         Scalar;
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

    virtual ~DampedWienerProcessModel() { }

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

#endif
