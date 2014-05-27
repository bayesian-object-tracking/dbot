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
#include <state_filtering/distribution/gaussian/gaussian_distribution.hpp>
#include <state_filtering/distribution/brownian/damped_brownian_motion.hpp>
#include <state_filtering/distribution/brownian/integrated_damped_brownian_motion.hpp>

#include <state_filtering/process_model/brownian_process_model.hpp>

namespace filter
{

void KalmanFilter::predict(const Estimate &prior_desc,
                           double delta_time,
                           Estimate &prediction_desc)
{    
    DampedBrownianMotion<double, 15> damped_brownian_motion;
    damped_brownian_motion.sample();

    IntegratedDampedBrownianMotion<double, 17> integrated_damped_brownian_motion;
    integrated_damped_brownian_motion.sample();

    GaussianDistribution<double, 13, 13> gaussian;
    GaussianDistribution<double, Eigen::Dynamic, Eigen::Dynamic> dynamic_gaussian(7);

    BrownianProcessModel<double, 13, 6, 13> brownian_process_model;
    brownian_process_model.sample();

//    //gaussian.setNormal();

//    std::cout << "gaussian.variableSize() = " << gaussian.variableSize() << std::endl;
//    std::cout << "dynamic_gaussian.variableSize() = " << dynamic_gaussian.variableSize() << std::endl;

//    std::cout << "gaussian.mean() = " << gaussian.mean().transpose() << std::endl;
//    std::cout << "gaussian.covariance() = " << gaussian.covariance() << std::endl;

//    std::cout << "dynamic_gaussian.mean() = " << dynamic_gaussian.mean().transpose() << std::endl;
//    std::cout << "dynamic_gaussian.covariance() = " << dynamic_gaussian.covariance() << std::endl;

//    gaussian.setNormal();
//    dynamic_gaussian.setNormal();

//    std::cout << "gaussian.mean() = " << gaussian.mean().transpose() << std::endl;
//    std::cout << "gaussian.covariance() = " << gaussian.covariance() << std::endl;

//    std::cout << "dynamic_gaussian.mean() = " << dynamic_gaussian.mean().transpose() << std::endl;
//    std::cout << "dynamic_gaussian.covariance() = " << dynamic_gaussian.covariance() << std::endl;

    std::cout << "gaussian.sample() = " << gaussian.sample().transpose() << std::endl;
    std::cout << "dynamic_gaussian.sample() = " << dynamic_gaussian.sample().transpose() << std::endl;
}

void KalmanFilter::update(const Measurement &measurement,
                          const Estimate &prediction_desc,
                          Estimate &posterior_desc)
{
}

}

int main(int nargs, char** vargs)
{
    filter::KalmanFilter kf;
    filter::Estimate temp;

    kf.predict(temp, 0, temp);

    return 0;
}
