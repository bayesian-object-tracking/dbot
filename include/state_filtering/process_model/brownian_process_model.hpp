/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California,
 *                     Karlsruhe Institute of Technology
 *    Jan Issac (jan.issac@gmail.com)
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
 * Max-Planck-Institute for Intelligent Systems, University of Southern California (USC),
 *   Karlsruhe Institute of Technology (KIT)
 */

#ifndef STATE_FILTERING_PROCESS_MODEL_BROWNIAN_PROCESS_MODEL_HPP
#define STATE_FILTERING_PROCESS_MODEL_BROWNIAN_PROCESS_MODEL_HPP

#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/distribution/brownian/damped_brownian_motion.hpp>
#include <state_filtering/distribution/brownian/integrated_damped_brownian_motion.hpp>

namespace filter
{

template <typename Traits>
class BrownianProcessModel:
        public StationaryProcessModel<Traits>
{
public:
    typedef IntegratedDampedBrownianMotionTraits<double, 3, 3> BrownianMotionTraits;

    virtual SampleType sample()
    {
        return mean_;
    }

    virtual SampleType mapFromGaussian(const SampleType& sample) const
    {
        return mean_ + L_ * sample;
    }

private:
    // conditionals
    double delta_time_;
    Eigen::Matrix<double, 3, 1> initial_linear_pose_;
    Eigen::Matrix<double, 4, 1> initial_angular_pose_;
    Eigen::Matrix<double, 4, 3> initial_quaternion_matrix_;
    Eigen::Matrix<double, 3, 1> initial_linear_velocity_;
    Eigen::Matrix<double, 3, 1> initial_angular_velocity_;

    Eigen::Matrix<double, 3, 1> linear_acceleration_control_;
    Eigen::Matrix<double, 3, 1> angular_acceleration_control_;

    // parameters
    Eigen::Matrix<double, 3, 1> rotation_center_;

    // distributions
    IntegratedDampedBrownianMotion<BrownianMotionTraits> delta_linear_pose_distribution_;
    IntegratedDampedBrownianMotion<BrownianMotionTraits> delta_angular_pose_distribution_;
    DampedBrownianMotion<BrownianMotionTraits> linear_velocity_distribution_;
    DampedBrownianMotion<BrownianMotionTraits> angular_velocity_distribution_;
};

}

#endif
