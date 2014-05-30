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

#ifndef STATE_FILTERING_DISTRIBUTION_IMPLEMENTATIONS_INTEGRATED_DAMPED_BROWNIAN_MOTION_HPP
#define STATE_FILTERING_DISTRIBUTION_IMPLEMENTATIONS_INTEGRATED_DAMPED_BROWNIAN_MOTION_HPP

// boost
#include <boost/math/special_functions/gamma.hpp>

// state_filtering
#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/features/gaussian_mappable.hpp>
#include <state_filtering/distribution/features/gaussian_sampleable.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>

namespace filter
{

template <typename ScalarType_, int SIZE>
class IntegratedDampedBrownianMotion:
        public GaussianMappable<ScalarType_, SIZE, SIZE>,
        public GaussianSampleable<GaussianMappable<ScalarType_, SIZE, SIZE> >
{
public: /* distribution traits */
    typedef GaussianMappable<ScalarType_, SIZE, SIZE> BaseType;

    typedef typename BaseType::ScalarType           ScalarType;
    typedef typename BaseType::VariableType         VariableType;
    typedef typename BaseType::RandomsType          RandomsType;
    typedef Eigen::Matrix<ScalarType, SIZE, SIZE>   CovarianceType;

public:

    IntegratedDampedBrownianMotion():
        distribution_()
    {
        DISABLE_IF_DYNAMIC_SIZE(VariableType);
    }

    IntegratedDampedBrownianMotion(int size):
        distribution_(size)
    {
        DISABLE_IF_FIXED_SIZE(VariableType);
    }

    virtual ~IntegratedDampedBrownianMotion() { }

    virtual VariableType mapFromGaussian(const RandomsType& sample) const
    {
        return distribution_.mapFromGaussian(sample);
    }

    virtual void conditionals(const double& delta_time,
                              const VariableType& state,
                              const VariableType& velocity,
                              const VariableType& acceleration)
    {
        // TODO this hack is necessary at the moment. the gaussian distribution cannot deal with
        // covariance matrices which are not full rank, which is the case for time equal to zero
        double bounded_delta_time = delta_time;
        if(bounded_delta_time < 0.00001) bounded_delta_time = 0.00001;

        distribution_.mean(Expectation(state, velocity, acceleration, bounded_delta_time));
        distribution_.covariance(Covariance(bounded_delta_time));

        n_variables_ = state.rows();
    }
    virtual void parameters(
            const double& damping,
            const CovarianceType& acceleration_covariance)
    {
        damping_ = damping;
        acceleration_covariance_ = acceleration_covariance;
    }

    virtual int variableSize() const
    {
        return distribution_.variableSize();
    }

    virtual int randomsSize() const
    {
        return variableSize();
    }

private:
    VariableType Expectation(const VariableType& state,
                           const VariableType& velocity,
                           const VariableType& acceleration,
                           const double& delta_time)
    {
        VariableType expectation;
        expectation = state +
                (exp(-damping_ * delta_time) + damping_*delta_time  - 1.0)/pow(damping_, 2)
                * acceleration + (1.0 - exp(-damping_*delta_time))/damping_  * velocity;

        if(!std::isfinite(expectation.norm()))
            expectation = state +
                    0.5*delta_time*delta_time*acceleration +
                    delta_time*velocity;

        return expectation;
    }

    CovarianceType Covariance(const double& delta_time)
    {
        // the first argument to the gamma function should be equal to zero, which would not cause
        // the gamma function to diverge as long as the second argument is not zero, which will not
        // be the case. boost however does not accept zero therefore we set it to a very small
        // value, which does not make a bit difference for any realistic delta_time
        double factor =
                (-1.0 + exp(-2.0*damping_*delta_time))/(8.0*pow(damping_, 3)) +
                (2.0 - exp(-2.0*damping_*delta_time))/(4.0*pow(damping_,2)) * delta_time +
                (-1.5 + gamma_ + boost::math::tgamma(0.00000000001, 2.0*damping_*delta_time) +
                 log(2.0*damping_*delta_time))/(2.0*damping_)*pow(delta_time, 2);
        if(!std::isfinite(factor))
            factor = 1.0/3.0 * pow(delta_time, 3);

        return factor * acceleration_covariance_;
    }

private:
    size_t n_variables_;
    // conditionals
    GaussianDistribution<ScalarType, SIZE> distribution_;
    // parameters
    double damping_;
    CovarianceType acceleration_covariance_;
    // euler-mascheroni constant
    static const double gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
