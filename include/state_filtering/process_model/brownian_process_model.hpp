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

#include <state_filtering/tools/helper_functions.hpp>
#include <state_filtering/system_states/full_rigid_body_system.hpp>
#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/distribution/implementations/damped_brownian_motion.hpp>
#include <state_filtering/distribution/implementations/integrated_damped_brownian_motion.hpp>

namespace filter
{

namespace internals
{
template <typename ScalarType_, bool IS_DYNAMIC>
struct BrownianProcessModelBase
{
    enum
    {
        VariableSize = 13,
        RandomsSize = 6,
        ControlSize = 6
    };

    typedef StationaryProcessModel<> Type;
};

template <typename ScalarType_>
struct BrownianProcessModelBase<ScalarType_, false>
{
    enum
    {
        VariableSize =BrownianProcessModelBase<ScalarType_, true>::VariableSize,
        RandomsSize = BrownianProcessModelBase<ScalarType_, true>::RandomsSize,
        ControlSize = BrownianProcessModelBase<ScalarType_, true>::ControlSize
    };

    typedef StationaryProcessModel<ScalarType_, VariableSize, RandomsSize, ControlSize> Type;
};
}

template <typename ScalarType_ = double, bool IS_DYNAMIC = true>
class BrownianProcessModel:
        public internals::BrownianProcessModelBase<ScalarType_, IS_DYNAMIC>::Type
{
public: /* model traits */
    typedef internals::BrownianProcessModelBase<ScalarType_, IS_DYNAMIC> Base;

    typedef typename Base::Type                 BaseType;
    typedef typename BaseType::ScalarType       ScalarType;
    typedef typename BaseType::VariableType     VariableType;
    typedef typename BaseType::CovarianceType   CovarianceType;
    typedef typename BaseType::RandomsType      RandomsType;
    typedef typename BaseType::ControlType      ControlType;

    typedef FullRigidBodySystem<1> StateType;
    typedef IntegratedDampedBrownianMotion<ScalarType, 3> AccelerationDistribution;
    typedef DampedBrownianMotion<ScalarType, 3> VelocityDistribution;

public:
    ~BrownianProcessModel() { }

    virtual VariableType mapNormal(const RandomsType& randoms) const
    {
        StateType state;
        state.position() = state_.get_position() + delta_position_.mapNormal(randoms.topRows(3));
        state.orientation() = state_.get_orientation() + quaternion_map_ * delta_orientation_.mapNormal(randoms.bottomRows(3));
        state.linear_velocity() = linear_velocity_.mapNormal(randoms.topRows(3));
        state.angular_velocity() = angular_velocity_.mapNormal(randoms.bottomRows(3));

        // renormalize quaternion
        state.orientation().normalize();

        // transform to external coordinate system
        state.linear_velocity() -= state.angular_velocity().cross(state.position());
        state.position() -= state.get_rotation_matrix()*rotation_center_;

        return state;
    }

    virtual void conditionals(const double& delta_time,
                              const VariableType& state,
                              const ControlType& control)
    {
        state_ = state;
        quaternion_map_ = hf::QuaternionMatrix(state_.orientation());

        // transform the state which is the pose and velocity with respecto to the origin into our internal representation,
        // which is the position and velocity of the rotation_center and the orientation and angular velocity around the center
        state_.position() += state_.get_rotation_matrix()*rotation_center_;
        state_.linear_velocity() += state_.angular_velocity().cross(state_.position());

        // todo: should controls change coordintes as well?
        linear_velocity_.conditionals(delta_time,
                                      state_.linear_velocity(),
                                      control.topRows(3));
        angular_velocity_.conditionals(delta_time,
                                       state_.angular_velocity(),
                                       control.bottomRows(3));
        delta_position_.conditionals(delta_time,
                                     Eigen::Vector3d::Zero(),
                                     state_.linear_velocity(),
                                     control.topRows(3));
        delta_orientation_.conditionals(delta_time,
                                        Eigen::Vector3d::Zero(),
                                        state_.angular_velocity(),
                                        control.bottomRows(3));
    }

    virtual void parameters(
                const Eigen::Matrix<ScalarType, 3, 1>& rotation_center,
                const double& damping,
                const typename AccelerationDistribution::CovarianceType& linear_acceleration_covariance,
                const typename VelocityDistribution::CovarianceType& angular_acceleration_covariance)
    {
        rotation_center_ = rotation_center;

        delta_position_.parameters(damping, linear_acceleration_covariance);
        delta_orientation_.parameters(damping, angular_acceleration_covariance);
        linear_velocity_.parameters(damping, linear_acceleration_covariance);
        angular_velocity_.parameters(damping, angular_acceleration_covariance);
    }

    virtual int variableSize() const
    {
        return Base::VariableSize;
    }
    virtual int randomsSize() const
    {
        return Base::RandomsSize;
    }
    virtual int controlSize() const
    {
        return Base::ControlSize;
    }

private:
    // conditionals
    StateType state_;
    Eigen::Matrix<ScalarType, 4, 3> quaternion_map_;

    // parameters
    Eigen::Matrix<ScalarType, 3, 1> rotation_center_;

    // distributions
    AccelerationDistribution delta_position_;
    AccelerationDistribution delta_orientation_;
    VelocityDistribution linear_velocity_;
    VelocityDistribution angular_velocity_;
};

}

#endif
