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

typedef Eigen::Matrix<double, 3, 1> State;

typedef ff::FactorizedUnscentedKalmanFilter<
                ProcessModelDummy<State>,
                ProcessModelDummy<State>,
                ObservationModelDummy<State> > Filter;

const size_t INV_DIMENSION = 14;
const size_t SUBSAMPLING_FACTOR = 8;
const size_t OBSERVATION_DIMENSION = (640*480)/(SUBSAMPLING_FACTOR*SUBSAMPLING_FACTOR);
const size_t INVERSION_ITERATIONS = OBSERVATION_DIMENSION * 30;

TEST(InversionTests, SMWInversion)
{
    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(15, 15);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0,   0, 14, 14);
    Eigen::MatrixXd B = cov.block(0,  14, 14,  1);
    Eigen::MatrixXd C = cov.block(14,  0, 1,  14);
    Eigen::MatrixXd D = cov.block(14, 14, 1,   1);

    Eigen::MatrixXd L_A;
    Eigen::MatrixXd L_B;
    Eigen::MatrixXd L_C;
    Eigen::MatrixXd L_D;

    Eigen::MatrixXd cov_inv = cov.inverse();
    Eigen::MatrixXd cov_smw_inv;

    Eigen::MatrixXd A_inv = A.inverse();
    filter.SMWInversion(A_inv, B, C, D, L_A, L_B, L_C, L_D, cov_smw_inv);

    EXPECT_TRUE(cov_smw_inv.isApprox(cov_inv));
}

TEST(InversionTests, fullMatrixInversionSpeed)
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
    cov = cov * cov.transpose();

    Eigen::MatrixXd cov_inv;

    std::clock_t start = std::clock();
    size_t number_of_inversions = 0;
    while ( (( std::clock() - start ) / (double) CLOCKS_PER_SEC) < 1.0 )
    {
        cov_inv = cov.inverse();
        number_of_inversions++;
    }

    std::cout << "fullMatrixInversionSpeed::number_of_inversions: "
              << number_of_inversions
              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
              << std::endl;

}

TEST(InversionTests, SMWMatrixInversionSpeed)
{
    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0, 0, INV_DIMENSION-1, INV_DIMENSION-1);
    Eigen::MatrixXd B = cov.block(0, INV_DIMENSION-1, INV_DIMENSION-1, 1);
    Eigen::MatrixXd C = cov.block(INV_DIMENSION-1, 0, 1, INV_DIMENSION-1);
    Eigen::MatrixXd D = cov.block(INV_DIMENSION-1, INV_DIMENSION-1, 1, 1);
    Eigen::MatrixXd A_inv = A.inverse();

    Eigen::MatrixXd L_A = Eigen::MatrixXd(INV_DIMENSION-1, INV_DIMENSION-1);
    Eigen::MatrixXd L_B = Eigen::MatrixXd(INV_DIMENSION-1, 1);
    Eigen::MatrixXd L_C = Eigen::MatrixXd(1, INV_DIMENSION-1);
    Eigen::MatrixXd L_D = Eigen::MatrixXd(1, 1);

    Eigen::MatrixXd cov_smw_inv;
    std::clock_t start = std::clock();
    size_t number_of_inversions = 0;
    while ( ((std::clock() - start) / (double) CLOCKS_PER_SEC) < 1.0 )
    {
        filter.SMWInversion(A_inv, B, C, D, L_A, L_B, L_C, L_D, cov_smw_inv);
        number_of_inversions++;
    }

    std::cout << "SMWMatrixInversionSpeed::number_of_inversions: "
              << number_of_inversions
              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
              << std::endl;
}

TEST(InversionTests, SMWBlockMatrixInversionSpeed)
{
    Filter filter(boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ProcessModelDummy<State> >(),
                  boost::make_shared<ObservationModelDummy<State> >());

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INV_DIMENSION, INV_DIMENSION);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0, 0, INV_DIMENSION-1, INV_DIMENSION-1);
    Eigen::MatrixXd B = cov.block(0, INV_DIMENSION-1, INV_DIMENSION-1, 1);
    Eigen::MatrixXd C = cov.block(INV_DIMENSION-1, 0, 1, INV_DIMENSION-1);
    Eigen::MatrixXd D = cov.block(INV_DIMENSION-1, INV_DIMENSION-1, 1, 1);
    Eigen::MatrixXd A_inv = A.inverse();

    Eigen::MatrixXd L_A = Eigen::MatrixXd(INV_DIMENSION-1, INV_DIMENSION-1);
    Eigen::MatrixXd L_B = Eigen::MatrixXd(INV_DIMENSION-1, 1);
    Eigen::MatrixXd L_C = Eigen::MatrixXd(1, INV_DIMENSION-1);
    Eigen::MatrixXd L_D = Eigen::MatrixXd(1, 1);

    std::clock_t start = std::clock();
    size_t number_of_inversions = 0;
    while ( (( std::clock() - start ) / (double) CLOCKS_PER_SEC) < 1.0 )
    {
        filter.SMWInversion(A_inv, B, C, D, L_A, L_B, L_C, L_D);
        number_of_inversions++;
    }

    std::cout << "SMWMatrixBlockInversionSpeed::number_of_inversions: "
              << number_of_inversions
              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
              << std::endl;
}
