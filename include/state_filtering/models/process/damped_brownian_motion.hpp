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

#ifndef STATE_FILTERING_gaussian_IMPLEMENTATIONS_BROWNIAN_DAMPED_BROWNIAN_MOTION_HPP
#define STATE_FILTERING_gaussian_IMPLEMENTATIONS_BROWNIAN_DAMPED_BROWNIAN_MOTION_HPP

#include <Eigen/Dense>

#include <boost/assert.hpp>

#include <state_filtering/models/process/stationary_process_model.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>

namespace distributions
{

template <int SIZE = -1, typename ScalarType_ = double>
struct DampedWienerProcessTypes
{
    typedef ScalarType_                                         ScalarType;
    typedef Eigen::Matrix<ScalarType, SIZE, 1>                  VectorType;
    typedef StationaryProcess<ScalarType, VectorType, SIZE>     StationaryProcessType;
    typedef typename StationaryProcessType::InputType             PerturbationType;
};



template <int SIZE = -1, typename ScalarType_ = double>
class DampedWienerProcess: public DampedWienerProcessTypes<SIZE, ScalarType_>::StationaryProcessType
{
public:
    typedef typename DampedWienerProcessTypes<SIZE, ScalarType_>::ScalarType         ScalarType;
    typedef typename DampedWienerProcessTypes<SIZE, ScalarType_>::VectorType         VectorType;
    typedef typename DampedWienerProcessTypes<SIZE, ScalarType_>::PerturbationType   InputType;
    typedef GaussianDistribution<ScalarType, SIZE>                          GaussianType;
    typedef typename GaussianType::OperatorType                                      OperatorType;

public:
    DampedWienerProcess(): gaussian_()
    {
        DISABLE_IF_DYNAMIC_SIZE(VectorType);
    }

    explicit DampedWienerProcess(int size): gaussian_(size)
    {
        DISABLE_IF_FIXED_SIZE(VectorType);
    }

    virtual ~DampedWienerProcess() { }

    virtual VectorType MapNormal(const InputType& sample) const
    {
        return gaussian_.MapNormal(sample);
    }

    virtual void Conditional(const ScalarType&       delta_time,
                             const VectorType&       state,
                             const InputType& control)
    {
        gaussian_.Mean(Mean(delta_time, state, control));
        gaussian_.Covariance(Covariance(delta_time));
    }

    virtual void parameters(const ScalarType& damping, const OperatorType& noise_covariance)
    {
        damping_ = damping;
        noise_covariance_ = noise_covariance;
    }

    virtual int variable_size() const
    {
        return gaussian_.variable_size();
    }

    virtual int InputDimension() const
    {
        return variable_size();
    }

    virtual int control_size() const
    {
        return variable_size();
    }

private:
    VectorType Mean(const ScalarType& delta_time,
                         const VectorType& state,
                         const InputType& control)
    {
        VectorType state_expectation =     (1.0 - exp(-damping_*delta_time)) / damping_ * control
                                              + exp(-damping_*delta_time)*state;

        // if the damping_ is too small, the result might be nan, we thus return the limit for damping_ -> 0
        if(!std::isfinite(state_expectation.norm()))
            state_expectation = state + delta_time * control;

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

    /** @brief euler-mascheroni constant */
    static const ScalarType gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
