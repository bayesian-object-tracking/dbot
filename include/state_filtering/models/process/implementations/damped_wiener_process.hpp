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

#ifndef DISTRIBUTIONS_IMPLEMENTATIONS_DAMPED_WIENER_PROCESS_HPP
#define DISTRIBUTIONS_IMPLEMENTATIONS_DAMPED_WIENER_PROCESS_HPP

#include <Eigen/Dense>

#include <boost/assert.hpp>

#include <state_filtering/models/process/features/stationary_process.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>

namespace distributions
{

template <typename ScalarType_, int DIMENSION_EIGEN>
struct DampedWienerProcessTypes
{
    typedef ScalarType_                                                 ScalarType;
    typedef Eigen::Matrix<ScalarType, DIMENSION_EIGEN, 1>               StateType;
    typedef Eigen::Matrix<ScalarType, DIMENSION_EIGEN, 1>               InputType;

    typedef StationaryProcess<ScalarType, StateType, InputType>        StationaryProcessType;
    typedef GaussianMappable<ScalarType, StateType, DIMENSION_EIGEN>   GaussianMappableType;

    typedef typename GaussianMappableType::NoiseType                    NoiseType;
};



template <typename ScalarType_ = double, int DIMENSION_EIGEN = -1>
class DampedWienerProcess: public DampedWienerProcessTypes<ScalarType_, DIMENSION_EIGEN>::StationaryProcessType,
                           public DampedWienerProcessTypes<ScalarType_, DIMENSION_EIGEN>::GaussianMappableType
{
public:
    typedef DampedWienerProcessTypes<ScalarType_, DIMENSION_EIGEN>  Types;
    typedef typename Types::ScalarType                              ScalarType;
    typedef typename Types::StateType                               StateType;
    typedef typename Types::InputType                               InputType;
    typedef typename Types::NoiseType                               NoiseType;
    typedef Gaussian<ScalarType, DIMENSION_EIGEN>                   GaussianType;
    typedef typename GaussianType::OperatorType                     OperatorType;

public:
    DampedWienerProcess()
    {
        DISABLE_IF_DYNAMIC_SIZE(StateType);
    }

    explicit DampedWienerProcess(const unsigned& dimension): Types::GaussianMappableType(dimension),
                                                             gaussian_(dimension)
    {
        DISABLE_IF_FIXED_SIZE(StateType);
    }

    virtual ~DampedWienerProcess() { }

    virtual StateType MapGaussian(const NoiseType& sample) const
    {
        return gaussian_.MapGaussian(sample);
    }

    virtual void Condition(const ScalarType&  delta_time,
                           const StateType&  state,
                           const InputType&   input)
    {
        gaussian_.Mean(Mean(delta_time, state, input));
        gaussian_.Covariance(Covariance(delta_time));
    }
    virtual void Condition(const ScalarType&  delta_time,
                           const StateType&  state)
    {
        Condition(delta_time, state, InputType::Zero(Dimension()));
    }

    virtual void Parameters(const ScalarType& damping,
                            const OperatorType& noise_covariance)
    {
        damping_ = damping;
        noise_covariance_ = noise_covariance;
    }

    virtual unsigned Dimension() const
    {
        return this->NoiseDimension(); // all dimensions are the same
    }

private:
    StateType Mean(const ScalarType& delta_time,
                    const StateType& state,
                    const InputType& input)
    {
        StateType state_expectation = (1.0 - exp(-damping_*delta_time)) / damping_ * input +
                                              exp(-damping_*delta_time)  * state;

        // if the damping_ is too small, the result might be nan, we thus return the limit for damping_ -> 0
        if(!std::isfinite(state_expectation.norm()))
            state_expectation = state + delta_time * input;

        return state_expectation;
    }

    OperatorType Covariance(const ScalarType& delta_time)
    {
        ScalarType factor = (1.0 - exp(-2.0*damping_*delta_time))/(2.0*damping_);
        if(!std::isfinite(factor))
            factor = delta_time;

        return factor * noise_covariance_;
    }

private:
    // conditional
    GaussianType gaussian_;

    // parameters
    ScalarType damping_;
    OperatorType noise_covariance_;

    // euler-mascheroni constant
    static const ScalarType gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
