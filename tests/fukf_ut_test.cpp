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
#include <vector>
#include <ctime>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>
#include <boost/make_shared.hpp>

#include "fukf_dummy_models.hpp"

class UnscentedTransformTest:
        public testing::Test
{
public:
    typedef Eigen::Matrix<double, 3, 1> State;

    typedef ff::FactorizedUnscentedKalmanFilter<
                    ProcessModelDummy<State>,
                    ProcessModelDummy<State>,
                    ObservationModelDummy<State> > Filter;

    UnscentedTransformTest():
        filter(Filter(boost::make_shared<ProcessModelDummy<State> >(),
                      boost::make_shared<ProcessModelDummy<State> >(),
                      boost::make_shared<ObservationModelDummy<State> >()))
    {

    }

protected:
    Filter filter;
};

const double EPSILON = 1.0e-12;

TEST_F(UnscentedTransformTest, weights)
{
    double w_0;
    double w_i;
    filter.ComputeWeights(11, w_0, w_i);

    EXPECT_DOUBLE_EQ(1., w_0 + 10.*w_i);
}

TEST_F(UnscentedTransformTest, unscentedTransformMeanRecovery)
{
    std::vector<Filter::SigmaPoints> X;
    State a = State::Ones() * 9;
    State a_recovered;
    State mean;

    Filter::StateDistribution::Cov_aa cov_aa;
    cov_aa.setIdentity();

    filter.ComputeSigmaPointPartitions({{a, cov_aa}}, X);

    a_recovered.setZero();
    for (size_t i = 0; i < X[0].cols(); ++i)
    {
        a_recovered += X[0].col(i);
    }
    a_recovered *= 1./double(X[0].cols());

    filter.Mean(X[0], mean);

    EXPECT_TRUE(a.isApprox(mean));
    EXPECT_TRUE(a.isApprox(a_recovered));
}


TEST_F(UnscentedTransformTest, unscentedTransformCovarianceRecovery)
{
    std::vector<Filter::SigmaPoints> X;
    State a = State::Ones() * 9;
    Filter::StateDistribution::Cov_aa cov_aa;
    cov_aa.setIdentity();
    State mean;

    filter.ComputeSigmaPointPartitions({{a, cov_aa}}, X);
    filter.Mean(X[0], mean);

    filter.Normalize(mean, X[0]);

    EXPECT_TRUE(cov_aa.isApprox(X[0]*X[0].transpose()));
}
