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
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include <boost/unordered_map.hpp>

#include <pose_tracking/models/observation_models/continuous_kinect_pixel_observation_model.hpp>

TEST(HashMapAndEigen, hasmapsForEigenMatrixXd)
{
    Eigen::MatrixXd someMatrix = Eigen::MatrixXd::Random(15, 1);
    Eigen::MatrixXd someOtherMatrix = Eigen::MatrixXd::Random(15, 1);
    Eigen::MatrixXd anotherMatrix = Eigen::MatrixXd::Random(15, 1);
    Eigen::MatrixXd sameMatrix = someMatrix;

    boost::unordered_map<Eigen::MatrixXd, int, boost::hash<Eigen::MatrixXd> > boolify_eigen_stuff;

    boolify_eigen_stuff[someMatrix] = 654654;
    boolify_eigen_stuff[someOtherMatrix] = 5;
    boolify_eigen_stuff[anotherMatrix] = 47;

    EXPECT_EQ(boolify_eigen_stuff[someMatrix], 654654);
    EXPECT_EQ(boolify_eigen_stuff[someOtherMatrix], 5);
    EXPECT_EQ(boolify_eigen_stuff[anotherMatrix], 47);
    EXPECT_EQ(boolify_eigen_stuff[sameMatrix], 654654);
}
