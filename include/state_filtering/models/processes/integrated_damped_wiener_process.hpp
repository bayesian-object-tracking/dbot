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

#ifndef DISTRIBUTIONS_IMPLEMENTATIONS_INTEGRATED_DAMPED_WIENER_PROCESS_HPP
#define DISTRIBUTIONS_IMPLEMENTATIONS_INTEGRATED_DAMPED_WIENER_PROCESS_HPP

// boost
#include <boost/math/special_functions/gamma.hpp>

// state_filtering

#include <state_filtering/distributions/features/gaussian_mappable.hpp>
#include <state_filtering/distributions/gaussian.hpp>
#include <state_filtering/models/processes/damped_wiener_process.hpp>

namespace sf
{

// Forward declarations
template <typename Scalar_, int INPUT_DIMENSION> class IntegratedDampedWienerProcess;

namespace internal
{
/**
 * IntegratedDampedWienerProcess distribution traits specialization
 * \internal
 */
template <typename Scalar_, int INPUT_DIMENSION>
struct Traits<IntegratedDampedWienerProcess<Scalar_, INPUT_DIMENSION> >
{
    enum
    {
        StateDimension = INPUT_DIMENSION == -1 ? -1 : 2 * INPUT_DIMENSION,
        InputDimension = INPUT_DIMENSION,
        NoiseDimension = INPUT_DIMENSION
    };

    typedef Scalar_                                    Scalar;
    typedef Eigen::Matrix<Scalar, StateDimension, 1>   State;
    typedef Eigen::Matrix<Scalar, InputDimension, 1>   Input;

    typedef StationaryProcessInterface<State, Input>   StationaryProcessInterfaceBase;
    typedef GaussianMappable<State, NoiseDimension>    GaussianMappableBase;

    typedef Eigen::Matrix<Scalar, InputDimension, 1>   WienerProcessState;
    typedef DampedWienerProcess<WienerProcessState>    DampedWienerProcessType;
    typedef Gaussian<Scalar, InputDimension>           GaussianType;

    typedef typename GaussianMappableBase::Noise       Noise;
    typedef typename GaussianType::Operator            Operator;
};
}

/**
 * \class IntegratedDampedWienerProcess
 *
 * \ingroup distributions
 * \ingroup process_models
 */
template <typename Scalar_ = double, int INPUT_DIMENSION = -1>
class IntegratedDampedWienerProcess:
        public internal::Traits<IntegratedDampedWienerProcess<Scalar_, INPUT_DIMENSION> >::StationaryProcessInterfaceBase,
        public internal::Traits<IntegratedDampedWienerProcess<Scalar_, INPUT_DIMENSION> >::GaussianMappableBase
{
public:
    typedef internal::Traits<IntegratedDampedWienerProcess<Scalar_, INPUT_DIMENSION> > Traits;

    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::State      State;
    typedef typename Traits::Input      Input;
    typedef typename Traits::Operator   Operator;
    typedef typename Traits::Noise      Noise;

    typedef typename Traits::GaussianType               GaussianType;
    typedef typename Traits::DampedWienerProcessType    DampedWienerProcessType;

public:
    IntegratedDampedWienerProcess()
    {
        SF_DISABLE_IF_DYNAMIC_SIZE(State);
    }

    IntegratedDampedWienerProcess(const unsigned& size):
        Traits::GaussianMappableBase(size),
        Traits::DampedWienerProcessType(size),
        position_distribution_(size)
    {
        SF_DISABLE_IF_FIXED_SIZE(State);
    }

    virtual ~IntegratedDampedWienerProcess() { }

    virtual State MapGaussian(const Noise& sample) const
    {
        State state(StateDimension());
        state.topRows(InputDimension())     = position_distribution_.MapGaussian(sample);
        state.bottomRows(InputDimension())  = velocity_distribution_.MapGaussian(sample);
        return state;
    }


    virtual void Condition(const Scalar&  delta_time,
                           const State&  state,
                           const Input&   input)
    {
        position_distribution_.Mean(Mean(state.topRows(InputDimension()), // position
                                                state.bottomRows(InputDimension()), // velocity
                                                input, // acceleration
                                                delta_time));
        position_distribution_.Covariance(Covariance(delta_time));

        velocity_distribution_.Condition(delta_time, state.bottomRows(InputDimension()), input);
    }
    virtual void Condition(const Scalar&  delta_time,
                           const State&  state)
    {
        Condition(delta_time, state, Input::Zero(InputDimension()));
    }

    virtual void Parameters(
            const double& damping,
            const Operator& acceleration_covariance)
    {
        damping_ = damping;
        acceleration_covariance_ = acceleration_covariance;

        velocity_distribution_.Parameters(damping, acceleration_covariance);
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
    Input Mean(const Input& state,
                   const Input& velocity,
                   const Input& acceleration,
                   const double& delta_time)
    {
        Input mean;
        mean = state +
                (exp(-damping_ * delta_time) + damping_*delta_time  - 1.0)/pow(damping_, 2)
                * acceleration + (1.0 - exp(-damping_*delta_time))/damping_  * velocity;

        if(!std::isfinite(mean.norm()))
            mean = state +
                    0.5*delta_time*delta_time*acceleration +
                    delta_time*velocity;

        return mean;
    }

    Operator Covariance(const Scalar& delta_time)
    {
        // the first argument to the gamma function should be equal to zero, which would not cause
        // the gamma function to diverge as long as the second argument is not zero, which will not
        // be the case. boost however does not accept zero therefore we set it to a very small
        // value, which does not make a bit difference for any realistic delta_time
        Scalar factor =
               (-1.0 + exp(-2.0*damping_*delta_time))/(8.0*pow(damping_, 3)) +
                (2.0 - exp(-2.0*damping_*delta_time))/(4.0*pow(damping_, 2)) * delta_time +
                (-1.5 + gamma_ + boost::math::tgamma(0.00000000001, 2.0*damping_*delta_time) +
                 log(2.0*damping_*delta_time))/(2.0*damping_)*pow(delta_time, 2);
        if(!std::isfinite(factor))
            factor = 1.0/3.0 * pow(delta_time, 3);

        return factor * acceleration_covariance_;
    }

private:
    DampedWienerProcessType velocity_distribution_;

    // conditionals
    GaussianType position_distribution_;

    // parameters
    Scalar damping_;
    Operator acceleration_covariance_;

    // euler-mascheroni constant
    static const Scalar gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
