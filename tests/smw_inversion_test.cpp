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

#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>

const double EPSILON = 1.0e-12;

template <typename State_>
class ProcessModelDummy:
        ff::StationaryProcessModel<State_, Eigen::Matrix<double, 0, 0> >
{
public:
    typedef State_ State;
    typedef Eigen::Matrix<double, 0, 0> Input;

    virtual void Condition(const double& delta_time,
                           const State& state,
                           const Input& input)
    {
        // foo
    }
};

template <typename State>
class ObservationModelDummy
{
public:
    virtual void predict(const State& state)
    {
        // foo
    }
};

typedef Eigen::Matrix<double, 3, 1> State;
typedef ff::FactorizedUnscentedKalmanFilter<
                ProcessModelDummy<State>,
                ProcessModelDummy<State>,
                ObservationModelDummy<State> > Filter;

TEST(InversionTests, SMWInversion)
{
    Filter filter;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(15, 15);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0,   0, 14, 14);
    Eigen::MatrixXd B = cov.block(0,  14, 14,  1);
    Eigen::MatrixXd C = cov.block(14,  0, 1,  14);
    Eigen::MatrixXd D = cov.block(14, 14, 1,   1);

    Eigen::MatrixXd cov_inv = cov.inverse();
    Eigen::MatrixXd cov_smw_inv;

    Eigen::MatrixXd A_inv = A.inverse();
    filter.SMWInversion(A_inv, B, C, D, cov_smw_inv);

    EXPECT_TRUE(cov_smw_inv.isApprox(cov_inv, EPSILON));
}

size_t INVERSION_DIMENSION = 14;
size_t SUBSAMPLING_FACTOR = 8;
size_t OBSERVATION_DIMENSION = (640*480)/(SUBSAMPLING_FACTOR*SUBSAMPLING_FACTOR);
size_t INVERSION_ITERATIONS = OBSERVATION_DIMENSION * 30;

TEST(InversionTests, fullMatrixInversionSpeed)
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INVERSION_DIMENSION, INVERSION_DIMENSION);
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
    Filter filter;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INVERSION_DIMENSION, INVERSION_DIMENSION);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0, 0, INVERSION_DIMENSION-1, INVERSION_DIMENSION-1);
    Eigen::MatrixXd B = cov.block(0, INVERSION_DIMENSION-1, INVERSION_DIMENSION-1, 1);
    Eigen::MatrixXd C = cov.block(INVERSION_DIMENSION-1, 0, 1, INVERSION_DIMENSION-1);
    Eigen::MatrixXd D = cov.block(INVERSION_DIMENSION-1, INVERSION_DIMENSION-1, 1, 1);
    Eigen::MatrixXd A_inv = A.inverse();

    Eigen::MatrixXd cov_smw_inv;
    std::clock_t start = std::clock();
    size_t number_of_inversions = 0;
    while ( (( std::clock() - start ) / (double) CLOCKS_PER_SEC) < 1.0 )
    {
        filter.SMWInversion(A_inv, B, C, D, cov_smw_inv);
        number_of_inversions++;
    }

    std::cout << "SMWMatrixInversionSpeed::number_of_inversions: "
              << number_of_inversions
              << "(" << number_of_inversions/OBSERVATION_DIMENSION << " fps)"
              << std::endl;
}

TEST(InversionTests, SMWBlockMatrixInversionSpeed)
{
    Filter filter;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Random(INVERSION_DIMENSION, INVERSION_DIMENSION);
    cov = cov * cov.transpose();

    Eigen::MatrixXd A = cov.block(0, 0, INVERSION_DIMENSION-1, INVERSION_DIMENSION-1);
    Eigen::MatrixXd B = cov.block(0, INVERSION_DIMENSION-1, INVERSION_DIMENSION-1, 1);
    Eigen::MatrixXd C = cov.block(INVERSION_DIMENSION-1, 0, 1, INVERSION_DIMENSION-1);
    Eigen::MatrixXd D = cov.block(INVERSION_DIMENSION-1, INVERSION_DIMENSION-1, 1, 1);
    Eigen::MatrixXd A_inv = A.inverse();

    Eigen::MatrixXd L_A = Eigen::MatrixXd(INVERSION_DIMENSION-1, INVERSION_DIMENSION-1);
    Eigen::MatrixXd L_B = Eigen::MatrixXd(INVERSION_DIMENSION-1, 1);
    Eigen::MatrixXd L_C = Eigen::MatrixXd(1, INVERSION_DIMENSION-1);
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

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
