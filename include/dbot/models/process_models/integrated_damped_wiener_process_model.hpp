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

#ifndef FAST_FILTERING_MODELS_PROCESS_MODELS_INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP
#define FAST_FILTERING_MODELS_PROCESS_MODELS_INTEGRATED_DAMPED_WIENER_PROCESS_MODEL_HPP

// boost
#include <boost/static_assert.hpp>
#include <boost/math/special_functions/gamma.hpp>

// state_filtering
#include <fl/util/assertions.hpp>
//#include <fast_filtering/distributions/interfaces/gaussian_map.hpp>
//#include <fast_filtering/distributions/gaussian.hpp>
#include <dbot/models/process_models/damped_wiener_process_model.hpp>
#include <fl/distribution/gaussian.hpp>

//TODO: THIS IS A LINEAR GAUSSIAN PROCESS, THIS CLASS SHOULD DISAPPEAR


namespace dbot
{

// Forward declarations
template <typename State> class IntegratedDampedWienerProcessModel;

namespace internal
{
/**
 * IntegratedDampedWienerProcess distribution traits specialization
 * \internal
 */
template <typename State_>
struct Traits<IntegratedDampedWienerProcessModel<State_> >
{
    enum
    {
        STATE_DIMENSION = State_::SizeAtCompileTime,
        DEGREE_OF_FREEDOM = STATE_DIMENSION != -1 ? STATE_DIMENSION/2 : -1
    };

    typedef State_                                      State;
    typedef typename State::Scalar                      Scalar;
    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> Input;
    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> Noise;

//    typedef StationaryProcessModel<State, Input>    ProcessModelBase;
//    typedef GaussianMap<State, Noise>     GaussianMapBase;

    typedef Eigen::Matrix<Scalar, DEGREE_OF_FREEDOM, 1> WienerProcessState;
    typedef DampedWienerProcessModel<WienerProcessState>     DampedWienerProcessType;
    typedef fl::Gaussian<Noise>                             GaussianType;

    typedef typename GaussianType::SecondMoment      Operator;
};
}

/**
 * \class IntegratedDampedWienerProcess
 *
 * \ingroup distributions
 * \ingroup process_models
 */
template <typename State_>
class IntegratedDampedWienerProcessModel
{
public:
    typedef internal::Traits<IntegratedDampedWienerProcessModel<State_> > Traits;

    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::State      State;
    typedef typename Traits::Input      Input;
    typedef typename Traits::Operator   Operator;
    typedef typename Traits::Noise      Noise;

    typedef typename Traits::GaussianType               GaussianType;
    typedef typename Traits::DampedWienerProcessType    DampedWienerProcessType;

    enum
    {
        STATE_DIMENSION = Traits::STATE_DIMENSION,
        DEGREE_OF_FREEDOM = Traits::DEGREE_OF_FREEDOM
    };

public:
    /// \todo uncomment default argument

    IntegratedDampedWienerProcessModel(
            const double& delta_time,
            const unsigned& degree_of_freedom = DEGREE_OF_FREEDOM):
        velocity_distribution_(delta_time, degree_of_freedom),
        position_distribution_(degree_of_freedom),
        delta_time_(delta_time)
    {
        std::cout << "delta_time " << delta_time_ << std::endl;
        static_assert_base(State, Eigen::Matrix<Scalar, STATE_DIMENSION, 1>);

        BOOST_STATIC_ASSERT_MSG(
                STATE_DIMENSION % 2 == 0 || STATE_DIMENSION == Eigen::Dynamic,
                "Dimension must be a multitude of 2");
    }

    virtual ~IntegratedDampedWienerProcessModel() noexcept { }

    virtual State MapStandardGaussian(const Noise& sample) const
    {
        State state(StateDimension());
        state.topRows(InputDimension())     = position_distribution_.map_standard_normal(sample);
        state.bottomRows(InputDimension())  = velocity_distribution_.MapStandardGaussian(sample);
        return state;
    }


    virtual void Condition(const State& state,
                           const Input& input)
    {
        position_distribution_.mean(Mean(state.topRows(InputDimension()), // position
                                                state.bottomRows(InputDimension()), // velocity
                                                input, // acceleration
                                                delta_time_));
        position_distribution_.covariance(Covariance(delta_time_));

        velocity_distribution_.Condition(state.bottomRows(InputDimension()), input);
    }


//    virtual void Condition(const Scalar&  delta_time,
//                           const State&  state,
//                           const Input&   input)
//    {
//        position_distribution_.mean(Mean(state.topRows(InputDimension()), // position
//                                                state.bottomRows(InputDimension()), // velocity
//                                                input, // acceleration
//                                                delta_time));
//        position_distribution_.covariance(Covariance(delta_time));

//        velocity_distribution_.Condition(delta_time, state.bottomRows(InputDimension()), input);
//    }





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

    virtual unsigned NoiseDimension() const
    {
        return velocity_distribution_.Dimension();
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

    double delta_time_;

    // euler-mascheroni constant
    static constexpr Scalar gamma_ = 0.57721566490153286060651209008240243104215933593992;
};

}

#endif
