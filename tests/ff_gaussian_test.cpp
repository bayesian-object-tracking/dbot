

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

#include <fast_filtering/distributions/gaussian.hpp>

class GaussianTests:
        public testing::Test
{
protected:
    template <typename Gaussian>
    void TestGaussianDimension(Gaussian& gaussian, size_t dim)
    {
        EXPECT_EQ(gaussian.Dimension(), dim);
        EXPECT_EQ(gaussian.NoiseDimension(), dim);
        EXPECT_EQ(gaussian.Mean().rows(), dim);
        EXPECT_EQ(gaussian.Covariance().rows(), dim);
        EXPECT_EQ(gaussian.Covariance().cols(), dim);

        typename Gaussian::Noise noise =
                Gaussian::Noise::Random(gaussian.NoiseDimension(),1);
        EXPECT_EQ(gaussian.MapStandardGaussian(noise).rows(), dim);
    }

    template <typename Gaussian>
    void TestGaussianDimensionAfterModification(Gaussian& gaussian,
                                                const size_t dim)
    {
        // test dimensions before modification
        TestGaussianDimension(gaussian, dim);

        // test dimensions after setting to standard gaussian
        gaussian.SetStandard();
        TestGaussianDimension(gaussian, dim);

        // test dimension after modifying the mean
        typename Gaussian::Vector random_mean
                = Gaussian::Vector::Random(dim, 1);
        gaussian.Mean(random_mean);
        TestGaussianDimension(gaussian, dim);

        // test dimension after setting a full ranked p.s.d. matrix
        typename Gaussian::Operator covariance;
        covariance = Gaussian::Operator::Random(dim, dim);
        covariance *= covariance.transpose();
        gaussian.Covariance(covariance);
        TestGaussianDimension(gaussian, dim);

        // test dimensions after setting a singular matrix
        covariance.row(0) = covariance.row(1);
        gaussian.Covariance(covariance);
        TestGaussianDimension(gaussian, dim);

        // test dimension after setting a full ranked diagonal matrix
        covariance = Gaussian::Operator::Identity(dim, dim)*1656;
        gaussian.DiagonalCovariance(covariance);
        TestGaussianDimension(gaussian, dim);

        // test dimension after setting a singular diagonal matrix
        covariance.row(0) = covariance.row(1);
        gaussian.DiagonalCovariance(covariance);
        TestGaussianDimension(gaussian, dim);
    }


    template <typename Gaussian>
    void TestRank(Gaussian& gaussian)
    {
        typename Gaussian::Vector random_mean
                = Gaussian::Vector::Random(gaussian.Dimension(), 1);

        typename Gaussian::Operator covariance;
        covariance = Gaussian::Operator::Random(
                    gaussian.Dimension(), gaussian.Dimension());

        // test rank effect before modification
        EXPECT_NE(gaussian.LogProbability(random_mean),
                  -std::numeric_limits<double>::infinity());

        // test rank effect after setting a a full ranked matrix
        covariance *= covariance.transpose();
        gaussian.Covariance(covariance);
        EXPECT_NE(gaussian.LogProbability(random_mean),
                  -std::numeric_limits<double>::infinity());

        // test rank effect after setting a singular matrix
        covariance.row(0) = covariance.row(1);
        gaussian.Covariance(covariance);
        EXPECT_EQ(gaussian.LogProbability(random_mean),
                  -std::numeric_limits<double>::infinity());


        // test rank effect after setting a a full ranked diagonal matrix
        covariance = Gaussian::Operator::Identity(
                    gaussian.Dimension(), gaussian.Dimension())*4235624;
        gaussian.DiagonalCovariance(covariance);
        EXPECT_NE(gaussian.LogProbability(random_mean),
                  -std::numeric_limits<double>::infinity());

        // test rank effect after setting a singular diagnoal matrix
        covariance.row(0) = covariance.row(1);
        gaussian.DiagonalCovariance(covariance);
        EXPECT_EQ(gaussian.LogProbability(random_mean),
                  -std::numeric_limits<double>::infinity());
    }
};

TEST_F(GaussianTests, fixedDimension)
{
    typedef Eigen::Matrix<double, 10, 1> Vector;
    ff::Gaussian<Vector> gaussian;

    TestGaussianDimensionAfterModification(gaussian, 10);
}

TEST_F(GaussianTests, dynamicDimension)
{
    const size_t dim = 10;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    ff::Gaussian<Vector> gaussian(dim);

    TestGaussianDimensionAfterModification(gaussian, dim);
}

TEST_F(GaussianTests, rankForFixedDimension)
{
    typedef Eigen::Matrix<double, 10, 1> Vector;
    ff::Gaussian<Vector> gaussian;

    TestRank(gaussian);
}

TEST_F(GaussianTests, rankForDynamicDimension)
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    ff::Gaussian<Vector> gaussian(10);

    TestRank(gaussian);
}
