/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Jan Issac (jan.issac@gmail.com)
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
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */


#ifndef MODELS_PROCESS_IMPLEMENTATIONS_BROWNIAN_OBJECT_MOTION_HPP
#define MODELS_PROCESS_IMPLEMENTATIONS_BROWNIAN_OBJECT_MOTION_HPP

#include <state_filtering/utils/helper_functions.hpp>
#include <state_filtering/states/floating_body_system.hpp>
#include <state_filtering/models/processes/features/stationary_process.hpp>
#include <state_filtering/models/processes/implementations/damped_wiener_process.hpp>
#include <state_filtering/models/processes/implementations/integrated_damped_wiener_process.hpp>

namespace sf
{

// Forward declarations
template <typename Scalar_, int OBJECTS> class BrownianObjectMotion;

namespace internal
{
/**
 * BrownianObjectMotion distribution traits specialization
 * \internal
 */
template <typename Scalar_, int OBJECTS>
struct Traits<BrownianObjectMotion<Scalar_, OBJECTS> >
{
    enum
    {
        DIMENSION_PER_OBJECT = 6,
        DIMENSION = (OBJECTS == -1) ?
                    -1 : OBJECTS * DIMENSION_PER_OBJECT,
        INPUT_DIMENSION = DIMENSION
    };

    typedef Scalar_                                     Scalar;
    typedef FloatingBodySystem<OBJECTS>                 State;
    typedef Eigen::Matrix<Scalar, INPUT_DIMENSION, 1>   Input;

    typedef Eigen::Quaternion<Scalar>                   Quaternion;
    typedef IntegratedDampedWienerProcess<Scalar, 3>    Process;

    typedef StationaryProcess<State, Input>             StationaryProcessBase;
    typedef GaussianMappable<State, DIMENSION>          GaussianMappableBase;

    typedef typename GaussianMappableBase::Noise        Noise;
};
}

/**
 * \class BrownianObjectMotion
 *
 * \ingroup distributions
 * \ingroup process_models
 */
template <typename Scalar_ = double, int OBJECTS = -1>
class BrownianObjectMotion:
        public internal::Traits<BrownianObjectMotion<Scalar_, OBJECTS> >::StationaryProcessBase,
        public internal::Traits<BrownianObjectMotion<Scalar_, OBJECTS> >::GaussianMappableBase
{
public:
    typedef internal::Traits<BrownianObjectMotion<Scalar_, OBJECTS> > Traits;

    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::State      State;
    typedef typename Traits::Input      Input;
    typedef typename Traits::Noise      Noise;
    typedef typename Traits::Quaternion Quaternion;
    typedef typename Traits::Process    Process;

    enum
    {
        DIMENSION            = Traits::DIMENSION,
        INPUT_DIMENSION      = Traits::INPUT_DIMENSION,
        DIMENSION_PER_OBJECT = Traits::DIMENSION_PER_OBJECT
    };

public:
    BrownianObjectMotion()
    {
        SF_DISABLE_IF_DYNAMIC_SIZE(State);

        quaternion_map_.resize(OBJECTS);
        rotation_center_.resize(OBJECTS);
        linear_process_.resize(OBJECTS);
        angular_process_.resize(OBJECTS);
    }

    BrownianObjectMotion(const unsigned& count_objects): Traits::GaussianMappableBase(count_objects*6),
                                                         state_(count_objects)
    {
        SF_DISABLE_IF_FIXED_SIZE(State);

        quaternion_map_.resize(count_objects);
        rotation_center_.resize(count_objects);
        linear_process_.resize(count_objects);
        angular_process_.resize(count_objects);
    }

    virtual ~BrownianObjectMotion() { }

    virtual State MapGaussian(const Noise& sample) const
    {
        State new_state(state_.bodies_size());
        for(size_t i = 0; i < new_state.bodies_size(); i++)
        {
            Eigen::Matrix<Scalar, 3, 1> position_noise    = sample.template middleRows<3>(i*DIMENSION_PER_OBJECT);
            Eigen::Matrix<Scalar, 3, 1> orientation_noise = sample.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3);
            Eigen::Matrix<Scalar, 6, 1> linear_delta      = linear_process_[i].MapGaussian(position_noise);
            Eigen::Matrix<Scalar, 6, 1> angular_delta     = angular_process_[i].MapGaussian(orientation_noise);

            new_state.position(i) = state_.position(i) + linear_delta.topRows(3);
            Quaternion updated_quaternion(state_.quaternion(i).coeffs() + quaternion_map_[i] * angular_delta.topRows(3));
            new_state.quaternion(updated_quaternion.normalized(), i);
            new_state.linear_velocity(i)  = linear_delta.bottomRows(3);
            new_state.angular_velocity(i) = angular_delta.bottomRows(3);

            // transform to external coordinate system
            new_state.linear_velocity(i) -= new_state.angular_velocity(i).cross(state_.position(i));
            new_state.position(i)        -= new_state.rotation_matrix(i)*rotation_center_[i];
        }

        return new_state;
    }

    virtual void Condition(const Scalar& delta_time,
                           const State&  state,
                           const Input&  control)
    {
        state_ = state;
        for(size_t i = 0; i < state_.bodies_size(); i++)
        {
            quaternion_map_[i] = hf::QuaternionMatrix(state_.quaternion(i).coeffs());

            // transform the state, which is the pose and velocity with respect to to the origin,
            // into internal representation, which is the position and velocity of the center
            // and the orientation and angular velocity around the center
            state_.position(i)        += state_.rotation_matrix(i)*rotation_center_[i];
            state_.linear_velocity(i) += state_.angular_velocity(i).cross(state_.position(i));

            Eigen::Matrix<Scalar, 6, 1> linear_state;
            linear_state.topRows(3) = Eigen::Vector3d::Zero();
            linear_state.bottomRows(3) = state_.linear_velocity(i);
            linear_process_[i].Condition(delta_time,
                                         linear_state,
                                         control.template middleRows<3>(i*DIMENSION_PER_OBJECT));

            Eigen::Matrix<Scalar, 6, 1> angular_state;
            angular_state.topRows(3) = Eigen::Vector3d::Zero();
            angular_state.bottomRows(3) = state_.angular_velocity(i);
            angular_process_[i].Condition(delta_time,
                                          angular_state,
                                          control.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3));
        }
    }
    virtual void Condition(const Scalar&  delta_time,
                           const State&  state)
    {
        Condition(delta_time, state, Input::Zero(InputDimension()));
    }


    virtual void Parameters(const size_t&                           object_index,
                            const Eigen::Matrix<Scalar, 3, 1>&  rotation_center,
                            const Scalar&                       damping,
                            const typename Process::Operator&   linear_acceleration_covariance,
                            const typename Process::Operator&   angular_acceleration_covariance)
    {
        rotation_center_[object_index] = rotation_center;
        linear_process_[object_index].Parameters(damping, linear_acceleration_covariance);
        angular_process_[object_index].Parameters(damping, angular_acceleration_covariance);
    }

    virtual unsigned InputDimension() const
    {
        return this->NoiseDimension();
    }


private:
    // conditionals
    State state_;
    std::vector<Eigen::Matrix<Scalar, 4, 3> > quaternion_map_;

    // parameters
    std::vector<Eigen::Matrix<Scalar, 3, 1> > rotation_center_;

    // processes
    std::vector<Process>   linear_process_;
    std::vector<Process>   angular_process_;
};

}

#endif
