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
        VariableSize = BrownianProcessModelBase<ScalarType_, true>::VariableSize,
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
    typedef typename BaseType::ControlType ControlType;

    typedef IntegratedDampedBrownianMotion<ScalarType, 3> AccelerationDistribution;
    typedef DampedBrownianMotion<ScalarType, 3> VelocityDistribution;

public:
    ~BrownianProcessModel() { }

    virtual VariableType mapFromGaussian(const RandomsType& randoms) const
    {
        VariableType state(variableSize());

        state.topRows(3) = initial_linear_pose_ +
                delta_linear_pose_distribution_.mapFromGaussian(randoms.topRows(3));
        state.middleRows(3, 4) = (initial_angular_pose_ +
                                  initial_quaternion_matrix_ * delta_angular_pose_distribution_.mapFromGaussian(randoms.bottomRows(3))).normalized();
        state.middleRows(7, 3) = linear_velocity_distribution_.mapFromGaussian(randoms.topRows(3));
        state.middleRows(10, 3) = angular_velocity_distribution_.mapFromGaussian(randoms.bottomRows(3));

        // transform to external representation
        state.middleRows(7, 3) -= state.template middleRows<3>(10).cross(state.template topRows<3>());
        state.topRows(3) -= Eigen::Quaterniond(state.template middleRows<4>(3)).toRotationMatrix()*rotation_center_;

        return state;
    }

    virtual void conditionals(
                const double& delta_time,
                const VariableType& state,
                const ControlType& control)
    {
        if(std::isfinite(delta_time))
            delta_time_ = delta_time;
        else
            delta_time_ = 0;
        // todo this hack is necessary at the moment because the gaussian distribution cannot deal with
        // covariance matrices which are not full rank, which is the case for time equal to zero
        if(delta_time_ < 0.00001) delta_time_ = 0.00001;


        initial_linear_pose_ = state.topRows(3);
        initial_angular_pose_ = state.middleRows(3, 4);
        initial_quaternion_matrix_ = hf::QuaternionMatrix(initial_angular_pose_);
        initial_linear_velocity_ = state.middleRows(7, 3);
        initial_angular_velocity_ = state.middleRows(10, 3);

        // we transform the state which is the pose and velocity with respecto to the origin into our internal representation,
        // which is the position and velocity of the rotation_center and the orientation and angular velocity around the center
        initial_linear_pose_ +=  Eigen::Quaterniond(initial_angular_pose_).toRotationMatrix()*rotation_center_;
        initial_linear_velocity_ += initial_angular_velocity_.cross(initial_linear_pose_);


        // todo: should these change coordintes as well?
        linear_acceleration_control_ = control.topRows(3);
        angular_acceleration_control_ = control.bottomRows(3);

        linear_velocity_distribution_.conditionals(delta_time_, initial_linear_velocity_, linear_acceleration_control_);
        angular_velocity_distribution_.conditionals(delta_time_, initial_angular_velocity_, angular_acceleration_control_);
        delta_linear_pose_distribution_.conditionals(delta_time_, Eigen::Vector3d::Zero(),
                                                     initial_linear_velocity_, linear_acceleration_control_);
        delta_angular_pose_distribution_.conditionals(delta_time_, Eigen::Vector3d::Zero(),
                                                      initial_angular_velocity_, angular_acceleration_control_);
    }

    virtual void parameters(
                const Eigen::Matrix<ScalarType, 3, 1>& rotation_center,
                const double& damping,
                const typename AccelerationDistribution::CovarianceType& linear_acceleration_covariance,
                const typename VelocityDistribution::CovarianceType& angular_acceleration_covariance)
    {
        rotation_center_ = rotation_center;

        delta_linear_pose_distribution_.parameters(damping, linear_acceleration_covariance);
        delta_angular_pose_distribution_.parameters(damping, angular_acceleration_covariance);
        linear_velocity_distribution_.parameters(damping, linear_acceleration_covariance);
        angular_velocity_distribution_.parameters(damping, angular_acceleration_covariance);
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
    double delta_time_;
    Eigen::Matrix<ScalarType, 3, 1> initial_linear_pose_;
    Eigen::Matrix<ScalarType, 4, 1> initial_angular_pose_;
    Eigen::Matrix<ScalarType, 4, 3> initial_quaternion_matrix_;
    Eigen::Matrix<ScalarType, 3, 1> initial_linear_velocity_;
    Eigen::Matrix<ScalarType, 3, 1> initial_angular_velocity_;

    Eigen::Matrix<ScalarType, 3, 1> linear_acceleration_control_;
    Eigen::Matrix<ScalarType, 3, 1> angular_acceleration_control_;

    // parameters
    Eigen::Matrix<ScalarType, 3, 1> rotation_center_;

    // distributions
    AccelerationDistribution delta_linear_pose_distribution_;
    AccelerationDistribution delta_angular_pose_distribution_;
    VelocityDistribution linear_velocity_distribution_;
    VelocityDistribution angular_velocity_distribution_;
};

}

#endif
