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
#include <state_filtering/models/process/features/stationary_process.hpp>
#include <state_filtering/models/process/implementations/damped_wiener_process.hpp>
#include <state_filtering/models/process/implementations/integrated_damped_wiener_process.hpp>

namespace distributions
{
template <typename ScalarType_, int OBJECTS_SIZE_EIGEN>
struct BrownianObjectMotionTypes
{
    enum
    {
        DIMENSION_PER_OBJECT = 6,
        DIMENSION = OBJECTS_SIZE_EIGEN == -1 ? -1 : OBJECTS_SIZE_EIGEN * DIMENSION_PER_OBJECT
    };

    typedef ScalarType_                                             ScalarType;
    typedef FloatingBodySystem<OBJECTS_SIZE_EIGEN>                  StateType;
    typedef Eigen::Matrix<ScalarType, DIMENSION, 1>                 InputType;
    typedef StationaryProcess<ScalarType, StateType, InputType>     StationaryProcessType;
    typedef GaussianMappable<ScalarType, StateType, DIMENSION>      GaussianMappableType;

    typedef typename GaussianMappableType::NoiseType                NoiseType;
};


template <typename ScalarType_ = double, int OBJECTS_SIZE_EIGEN = -1>
class BrownianObjectMotion: public BrownianObjectMotionTypes<ScalarType_, OBJECTS_SIZE_EIGEN>::StationaryProcessType,
                            public BrownianObjectMotionTypes<ScalarType_, OBJECTS_SIZE_EIGEN>::GaussianMappableType

{
public:
    // types from parents
    typedef BrownianObjectMotionTypes<ScalarType_, OBJECTS_SIZE_EIGEN>   Types;
    typedef typename Types::ScalarType                             ScalarType;
    typedef typename Types::StateType                              StateType;
    typedef typename Types::InputType                              InputType;
    typedef typename Types::NoiseType                              NoiseType;

    // new types
    typedef typename Eigen::Quaternion<ScalarType>                 Quaternion;
    typedef IntegratedDampedWienerProcess<ScalarType, 3>           Process;

    enum
    {
        DIMENSION_PER_OBJECT = Types::DIMENSION_PER_OBJECT
    };

public:
    BrownianObjectMotion()
    {
        DISABLE_IF_DYNAMIC_SIZE(StateType);

        quaternion_map_.resize(OBJECTS_SIZE_EIGEN);
        rotation_center_.resize(OBJECTS_SIZE_EIGEN);
        linear_process_.resize(OBJECTS_SIZE_EIGEN);
        angular_process_.resize(OBJECTS_SIZE_EIGEN);
    }

    BrownianObjectMotion(const unsigned& count_objects): Types::GaussianMappableType(count_objects*6),
                                                         state_(count_objects)
    {
        DISABLE_IF_FIXED_SIZE(StateType);

        quaternion_map_.resize(count_objects);
        rotation_center_.resize(count_objects);
        linear_process_.resize(count_objects);
        angular_process_.resize(count_objects);
    }

    virtual ~BrownianObjectMotion() { }

    virtual StateType MapGaussian(const NoiseType& sample) const
    {
        StateType new_state(state_.bodies_size());
        for(size_t i = 0; i < new_state.bodies_size(); i++)
        {
            Eigen::Matrix<ScalarType, 3, 1> position_noise    = sample.template middleRows<3>(i*DIMENSION_PER_OBJECT);
            Eigen::Matrix<ScalarType, 3, 1> orientation_noise = sample.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3);
            Eigen::Matrix<ScalarType, 6, 1> linear_delta      = linear_process_[i].MapGaussian(position_noise);
            Eigen::Matrix<ScalarType, 6, 1> angular_delta     = angular_process_[i].MapGaussian(orientation_noise);

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

    virtual void Condition(const ScalarType& delta_time,
                           const StateType&  state,
                           const InputType&  control)
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

            Eigen::Matrix<ScalarType, 6, 1> linear_state;
            linear_state.topRows(3) = Eigen::Vector3d::Zero();
            linear_state.bottomRows(3) = state_.linear_velocity(i);
            linear_process_[i].Condition(delta_time,
                                         linear_state,
                                         control.template middleRows<3>(i*DIMENSION_PER_OBJECT));

            Eigen::Matrix<ScalarType, 6, 1> angular_state;
            angular_state.topRows(3) = Eigen::Vector3d::Zero();
            angular_state.bottomRows(3) = state_.angular_velocity(i);
            angular_process_[i].Condition(delta_time,
                                          angular_state,
                                          control.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3));
        }
    }
    virtual void Condition(const ScalarType&  delta_time,
                           const StateType&  state)
    {
        Condition(delta_time, state, InputType::Zero(InputDimension()));
    }


    virtual void Parameters(const size_t&                           object_index,
                            const Eigen::Matrix<ScalarType, 3, 1>&  rotation_center,
                            const ScalarType&                       damping,
                            const typename Process::OperatorType&   linear_acceleration_covariance,
                            const typename Process::OperatorType&   angular_acceleration_covariance)
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
    StateType state_;
    std::vector<Eigen::Matrix<ScalarType, 4, 3> > quaternion_map_;

    // parameters
    std::vector<Eigen::Matrix<ScalarType, 3, 1> > rotation_center_;

    // processes
    std::vector<Process>   linear_process_;
    std::vector<Process>   angular_process_;
};

}

#endif
