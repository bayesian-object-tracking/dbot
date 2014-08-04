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


#ifndef STATE_FILTERING_PROCESS_MODEL_BROWNIAN_PROCESS_MODEL_HPP
#define STATE_FILTERING_PROCESS_MODEL_BROWNIAN_PROCESS_MODEL_HPP

#include <state_filtering/utils/helper_functions.hpp>
#include <state_filtering/states/floating_body_system.hpp>
#include <state_filtering/models/process/features/stationary_process.hpp>
#include <state_filtering/models/process/implementations/damped_wiener_process.hpp>
#include <state_filtering/models/process/implementations/integrated_damped_wiener_process.hpp>

namespace distributions
{
template <int SIZE_OBJECTS, typename ScalarType_>
struct BrownianObjectMotionTypes
{
    enum
    {
        DIMENSION_PER_OBJECT = 6,
        DIMENSION = SIZE_OBJECTS == -1 ? -1 : SIZE_OBJECTS * DIMENSION_PER_OBJECT
    };

    typedef ScalarType_                                             ScalarType;
    typedef FloatingBodySystem<SIZE_OBJECTS>                        VectorType;
    typedef Eigen::Matrix<ScalarType, DIMENSION, 1>                 InputType;
    typedef StationaryProcess<ScalarType, VectorType, InputType>    StationaryProcessType;
    typedef GaussianMappable<ScalarType, VectorType, DIMENSION>     GaussianMappableType;

    typedef typename GaussianMappableType::NoiseType                NoiseType;
};


template <int SIZE_OBJECTS = -1, typename ScalarType_ = double>
class BrownianObjectMotion: public BrownianObjectMotionTypes<SIZE_OBJECTS, ScalarType_>::StationaryProcessType,
                            public BrownianObjectMotionTypes<SIZE_OBJECTS, ScalarType_>::GaussianMappableType

{
public:
    // types from parents
    typedef BrownianObjectMotionTypes<SIZE_OBJECTS, ScalarType_>   Types;
    typedef typename Types::ScalarType                             ScalarType;
    typedef typename Types::VectorType                             VectorType;
    typedef typename Types::NoiseType                              NoiseType;

    // new types
    typedef typename Eigen::Quaternion<ScalarType>                 Quaternion;
    typedef IntegratedDampedWienerProcess<ScalarType, 3>           PositionDistribution;
    typedef DampedWienerProcess<ScalarType, 3>                     VelocityDistribution;

    enum
    {
        DIMENSION_PER_OBJECT = Types::DIMENSION_PER_OBJECT
    };

public:
    BrownianObjectMotion()
    {
        // todo: check whether this complains for dynamic size

        quaternion_map_.resize(SIZE_OBJECTS);
        rotation_center_.resize(SIZE_OBJECTS);
        delta_position_.resize(SIZE_OBJECTS);
        delta_orientation_.resize(SIZE_OBJECTS);
        linear_velocity_.resize(SIZE_OBJECTS);
        angular_velocity_.resize(SIZE_OBJECTS);
    }

    BrownianObjectMotion(const unsigned& count_objects):
        BrownianObjectMotionTypes<SIZE_OBJECTS, ScalarType_>::GaussianMappableType(count_objects*6),
        state_(count_objects)
    {

        quaternion_map_.resize(count_objects);
        rotation_center_.resize(count_objects);
        delta_position_.resize(count_objects);
        delta_orientation_.resize(count_objects);
        linear_velocity_.resize(count_objects);
        angular_velocity_.resize(count_objects);
    }

    virtual ~BrownianObjectMotion() { }

    virtual VectorType MapGaussian(const NoiseType& sample) const
    {
        VectorType new_state = state_;
        for(size_t i = 0; i < new_state.bodies_size(); i++)
        {
            new_state.position(i) = new_state.position(i) +
                    delta_position_[i].MapGaussian(sample.template middleRows<3>(i*DIMENSION_PER_OBJECT)).topRows(3);
            Quaternion updated_quaternion(new_state.quaternion(i).coeffs()
                       + quaternion_map_[i] *
                       delta_orientation_[i].MapGaussian(sample.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3)).topRows(3));
            new_state.quaternion(updated_quaternion.normalized(), i);
            new_state.linear_velocity(i) = linear_velocity_[i].MapGaussian(sample.template middleRows<3>(i*DIMENSION_PER_OBJECT));
            new_state.angular_velocity(i) = angular_velocity_[i].MapGaussian(sample.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3));

            // transform to external coordinate system
            new_state.linear_velocity(i) -= new_state.angular_velocity(i).cross(state_.position(i));
            new_state.position(i) -= new_state.rotation_matrix(i)*rotation_center_[i];
        }

        return new_state;
    }

    virtual void Condition(const ScalarType&  delta_time,
                           const VectorType&  state,
                           const NoiseType&   control)
    {
        state_ = state;

        for(size_t i = 0; i < state_.bodies_size(); i++)
        {
            quaternion_map_[i] = hf::QuaternionMatrix(state_.quaternion(i).coeffs());

            // transform the state which is the pose and velocity with respecto to the origin into our internal representation,
            // which is the position and velocity of the rotation_center and the orientation and angular velocity around the center
            state_.position(i) += state_.rotation_matrix(i)*rotation_center_[i];
            state_.linear_velocity(i) += state_.angular_velocity(i).cross(state_.position(i));

            // todo: should controls change coordintes as well?
            linear_velocity_[i].Condition( delta_time,
                                             state_.linear_velocity(i),
                                             control.template middleRows<3>(i*DIMENSION_PER_OBJECT));
            angular_velocity_[i].Condition( delta_time,
                                           state_.angular_velocity(i),
                                           control.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3));

            Eigen::Matrix<ScalarType, 6, 1> linear_state;
            linear_state.topRows(3) = Eigen::Vector3d::Zero();
            linear_state.bottomRows(3) = state_.linear_velocity(i);
            delta_position_[i].Condition(delta_time,
                                            linear_state,
                                         control.template middleRows<3>(i*DIMENSION_PER_OBJECT));


            Eigen::Matrix<ScalarType, 6, 1> angular_state;
            angular_state.topRows(3) = Eigen::Vector3d::Zero();
            angular_state.bottomRows(3) = state_.angular_velocity(i);
            delta_orientation_[i].Condition(delta_time,
                                            angular_state,
                                            control.template middleRows<3>(i*DIMENSION_PER_OBJECT + 3));
        }

    }


    virtual void parameters(const size_t&                                       object_index,
                            const Eigen::Matrix<ScalarType, 3, 1>&              rotation_center,
                            const ScalarType&                                   damping,
                            const typename PositionDistribution::OperatorType&  linear_acceleration_covariance,
                            const typename VelocityDistribution::OperatorType&  angular_acceleration_covariance)
    {
        rotation_center_[object_index] = rotation_center;

        delta_position_[object_index].Parameters(damping, linear_acceleration_covariance);
        delta_orientation_[object_index].Parameters(damping, angular_acceleration_covariance);
        linear_velocity_[object_index].Parameters(damping, linear_acceleration_covariance);
        angular_velocity_[object_index].Parameters(damping, angular_acceleration_covariance);
    }

private:
    // conditionals
    VectorType state_;
    std::vector<Eigen::Matrix<ScalarType, 4, 3> > quaternion_map_;

    // parameters
    std::vector<Eigen::Matrix<ScalarType, 3, 1> > rotation_center_;

    // distributions
    std::vector<PositionDistribution>   delta_position_;
    std::vector<PositionDistribution>   delta_orientation_;
    std::vector<VelocityDistribution>       linear_velocity_;
    std::vector<VelocityDistribution>       angular_velocity_;
};

}

#endif
