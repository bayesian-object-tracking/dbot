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

#ifndef STATE_FILTERING_gaussian_IMPLEMENTATIONS_INTEGRATED_DAMPED_BROWNIAN_MOTION_HPP
#define STATE_FILTERING_gaussian_IMPLEMENTATIONS_INTEGRATED_DAMPED_BROWNIAN_MOTION_HPP

// boost
#include <boost/math/special_functions/gamma.hpp>

// state_filtering
#include <state_filtering/distributions/distribution.hpp>
#include <state_filtering/distributions/features/gaussian_mappable.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>
#include <state_filtering/models/process/damped_brownian_motion.hpp>

namespace distributions
{

template <typename ScalarType_, int INPUT_DIM_EIGEN>
struct IntegratedDampedWienerProcessTypes
{
    typedef ScalarType_                                                                     ScalarType;
    typedef Eigen::Matrix<ScalarType, INPUT_DIM_EIGEN == -1 ? -1 : 2 * INPUT_DIM_EIGEN, 1>  VectorType;
    typedef Eigen::Matrix<ScalarType, INPUT_DIM_EIGEN, 1>                                   InputType;

    typedef StationaryProcess<ScalarType, VectorType, InputType>                            StationaryProcessType;
    typedef GaussianMappable<ScalarType, VectorType, INPUT_DIM_EIGEN>                       GaussianMappableType;

    typedef typename GaussianMappableType::NoiseType                                        NoiseType;

    typedef DampedWienerProcess<ScalarType, INPUT_DIM_EIGEN>                                DampedWienerProcessType;
};



template <typename ScalarType_ = double, int INPUT_DIM_EIGEN = -1>
class IntegratedDampedWienerProcess:
        public IntegratedDampedWienerProcessTypes<ScalarType_, INPUT_DIM_EIGEN>::StationaryProcessType,
        public IntegratedDampedWienerProcessTypes<ScalarType_, INPUT_DIM_EIGEN>::GaussianMappableType
{
public:
    typedef IntegratedDampedWienerProcessTypes<ScalarType_, INPUT_DIM_EIGEN>    Types;
    typedef typename Types::ScalarType                                          ScalarType;
    typedef typename Types::VectorType                                          VectorType;
    typedef typename Types::InputType                                           InputType;
    typedef typename Types::NoiseType                                           NoiseType;
    typedef typename Types::DampedWienerProcessType                             DampedWienerProcessType;
    typedef Gaussian<ScalarType, INPUT_DIM_EIGEN>                               GaussianType;
    typedef typename GaussianType::OperatorType                                 OperatorType;

public:
    IntegratedDampedWienerProcess()
    {
        DISABLE_IF_DYNAMIC_SIZE(VectorType);
    }

    IntegratedDampedWienerProcess(const unsigned& size): Types::GaussianMappableType(size),
                                                         Types::DampedWienerProcessType(size),
                                                         gaussian_(size)
    {
        DISABLE_IF_FIXED_SIZE(VectorType);
    }

    virtual ~IntegratedDampedWienerProcess() { }

    virtual VectorType MapGaussian(const NoiseType& sample) const
    {
        VectorType state(StateDimension());
        state.topRows(InputDimension())     = gaussian_.MapGaussian(sample);
        state.bottomRows(InputDimension())  = damped_wiener_process_.MapGaussian(sample);
        return state;
    }


    virtual void Condition(const ScalarType&  delta_time,
                           const VectorType&  state,
                           const InputType&   input)
    {
        gaussian_.Mean(Expectation(state.topRows(InputDimension()),
                                   state.bottomRows(InputDimension()),
                                   input,
                                   delta_time));
        gaussian_.Covariance(Covariance(delta_time));

        damped_wiener_process_.Condition(delta_time, state.bottomRows(InputDimension()), input);
    }

//    // THIS SHOULD GO AWAY!!
//    virtual void conditionals(const ScalarType& delta_time,
//                              const VectorType& state,
//                              const VectorType& velocity,
//                              const VectorType& acceleration)
//    {
//        gaussian_.Mean(Expectation(state, velocity, acceleration, delta_time));
//        gaussian_.Covariance(Covariance(delta_time));
//    }
    virtual void Parameters(
            const double& damping,
            const OperatorType& acceleration_covariance)
    {
        damping_ = damping;
        acceleration_covariance_ = acceleration_covariance;

        damped_wiener_process_.Parameters(damping, acceleration_covariance);
    }

    virtual unsigned InputDimension() const
    {
        return this->NoiseDimension();
    }

    virtual unsigned StateDimension() const
    {
        return this->NoiseDimension() * 2;
    }

private:
    InputType Expectation(const InputType& state,
                           const InputType& velocity,
                           const InputType& acceleration,
                           const double& delta_time)
    {
        InputType expectation;
        expectation = state +
                (exp(-damping_ * delta_time) + damping_*delta_time  - 1.0)/pow(damping_, 2)
                * acceleration + (1.0 - exp(-damping_*delta_time))/damping_  * velocity;

        if(!std::isfinite(expectation.norm()))
            expectation = state +
                    0.5*delta_time*delta_time*acceleration +
                    delta_time*velocity;

        return expectation;
    }

    OperatorType Covariance(const ScalarType& delta_time)
    {
        // the first argument to the gamma function should be equal to zero, which would not cause
        // the gamma function to diverge as long as the second argument is not zero, which will not
        // be the case. boost however does not accept zero therefore we set it to a very small
        // value, which does not make a bit difference for any realistic delta_time
        ScalarType factor =
               (-1.0 + exp(-2.0*damping_*delta_time))/(8.0*pow(damping_, 3)) +
                (2.0 - exp(-2.0*damping_*delta_time))/(4.0*pow(damping_, 2)) * delta_time +
                (-1.5 + gamma_ + boost::math::tgamma(0.00000000001, 2.0*damping_*delta_time) +
                 log(2.0*damping_*delta_time))/(2.0*damping_)*pow(delta_time, 2);
        if(!std::isfinite(factor))
            factor = 1.0/3.0 * pow(delta_time, 3);

        return factor * acceleration_covariance_;
    }

private:
    DampedWienerProcessType damped_wiener_process_;

    // conditionals
    GaussianType gaussian_;

    // parameters
    ScalarType damping_;
    OperatorType acceleration_covariance_;

    // euler-mascheroni constant
    static const double gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
