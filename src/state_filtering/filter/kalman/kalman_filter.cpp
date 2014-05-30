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
 * @date 05/19/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California (USC),
 *   Karlsruhe Institute of Technology (KIT)
 */

#include <Eigen/Dense>

#include <boost/static_assert.hpp>

#include <state_filtering/filter/kalman/kalman_filter.hpp>

#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>
#include <state_filtering/distribution/implementations/damped_brownian_motion.hpp>
#include <state_filtering/distribution/implementations/integrated_damped_brownian_motion.hpp>

#include <state_filtering/process_model/brownian_process_model.hpp>

namespace filter
{

void KalmanFilter::predict(const Estimate &prior_desc,
                           double delta_time,
                           Estimate &prediction_desc)
{    
//    GaussianDistribution<double, 13> gaussian;
//    DampedBrownianMotion<double, 15> damped_brownian_motion;
//    IntegratedDampedBrownianMotion<double, 17> integrated_damped_brownian_motion;
//    BrownianProcessModel<double> brownian_process_model;

//    std::cout << "gaussian.sample() = " << gaussian.sample().transpose() << std::endl;
//    std::cout << "brownian_process_model.sample() = " << brownian_process_model.sample().transpose() << std::endl;
//    std::cout << "damped_brownian_motion.sample() = " << damped_brownian_motion.sample().transpose() << std::endl;
//    std::cout << "integrated_damped_brownian_motion.sample() = " << integrated_damped_brownian_motion.sample().transpose() << std::endl;

//    GaussianDistribution<double, Eigen::Dynamic> dynamic_gaussian(6);
//    DampedBrownianMotion<double, Eigen::Dynamic> dynamic_damped_brownian_motion(6);
//    IntegratedDampedBrownianMotion<double, Eigen::Dynamic> dynamic_integrated_damped_brownian_motion(5);

//    std::cout << "dynamic_gaussian.sample() = " << dynamic_gaussian.sample().transpose() << std::endl;
//    std::cout << "dynamic_damped_brownian_motion.sample() = " << dynamic_damped_brownian_motion.sample().transpose() << std::endl;
//    std::cout << "dynamic_integrated_damped_brownian_motion.sample() = " << dynamic_integrated_damped_brownian_motion.sample().transpose() << std::endl;
}

void KalmanFilter::update(const Measurement &measurement,
                          const Estimate &prediction_desc,
                          Estimate &posterior_desc)
{
}

}

//int main(int nargs, char** vargs)
//{
//    filter::KalmanFilter kf;
//    filter::Estimate temp;

//    kf.predict(temp, 0, temp);

//    return 0;
//}
