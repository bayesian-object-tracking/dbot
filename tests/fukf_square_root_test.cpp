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
 *  LIABILITY, OR TOR-+T (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
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
#include <vector>
#include <ctime>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>
#include <boost/make_shared.hpp>

#include "fukf_dummy_models.hpp"

const double EPSILON = 1.0e-12;

typedef Eigen::Matrix<double, 3, 1> State;

typedef ff::FactorizedUnscentedKalmanFilter<
                ProcessModelDummy<State>,
                ProcessModelDummy<State>,
                ObservationModelDummy<State> > Filter;

const size_t DIMENSION = 10;

TEST(SquareRootTests, dense)
{
    Eigen::MatrixXd cov_sqrt;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(DIMENSION, DIMENSION);
    cov = cov * cov.transpose();

    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());
    filter.SquareRoot(cov, cov_sqrt);

    EXPECT_EQ(cov.rows(), cov_sqrt.rows());
    EXPECT_EQ(cov.cols(), cov_sqrt.cols());
    EXPECT_TRUE(cov.isApprox(cov_sqrt * cov_sqrt.transpose(), EPSILON));
}

TEST(SquareRootTests, diagonal)
{
    Eigen::MatrixXd cov_sqrt;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(DIMENSION, DIMENSION) * 1.3224;

    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());
    filter.SquareRootDiagonal(cov, cov_sqrt);

    EXPECT_EQ(cov.rows(), cov_sqrt.rows());
    EXPECT_EQ(cov.cols(), cov_sqrt.cols());
    EXPECT_TRUE(cov.isApprox(cov_sqrt * cov_sqrt.transpose(), EPSILON));
}

TEST(SquareRootTests, diagonalAsVector)
{
    Eigen::MatrixXd cov_sqrt;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Ones(DIMENSION, 1) * 1.3224;

    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());
    filter.SquareRootDiagonalAsVector(cov, cov_sqrt);

    EXPECT_EQ(cov.rows(), cov_sqrt.rows());
    EXPECT_EQ(1, cov_sqrt.cols());
    EXPECT_TRUE(cov.isApprox(cov_sqrt.cwiseProduct(cov_sqrt), EPSILON));
}

