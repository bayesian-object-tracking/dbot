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

#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>

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


/**
 * @brief PartitionedUnscentedTransformTest fixture
 */
class PartitionedUnscentedTransformTest:
        public testing::Test
{
public:
    typedef Eigen::Matrix<double, 3, 1> State;
    typedef ff::FactorizedUnscentedKalmanFilter<
                    ProcessModelDummy<State>,
                    ProcessModelDummy<State>,
                    ObservationModelDummy<State> > Filter;

    virtual void SetUp()
    {
        state.state_a_ = State::Random(3, 1);
        state.cov_aa_ = Filter::StateDistribution::CovAA::Random(3, 3);
        state.cov_aa_ = state.cov_aa_ * state.cov_aa_.transpose();
    }

protected:
    Filter filter;
    Filter::StateDistribution state;
};


/**
 * Test the offset behavior
 */
TEST_F(PartitionedUnscentedTransformTest, firstUTColumn)
{
    size_t joint_dimension = state.CohesiveStatesDimension() + 3;
    size_t number_of_points = 2 * joint_dimension + 1;

    Filter::SigmaPoints X(state.CohesiveStatesDimension(), number_of_points);
    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 0, X);

    EXPECT_TRUE(X.col(0).isApprox(state.state_a_));
}


/**
 * Test the offset behavior
 */
TEST_F(PartitionedUnscentedTransformTest, partitionOffset)
{
    size_t max_offset = 3;
    size_t joint_dimension = state.CohesiveStatesDimension() + max_offset;
    size_t number_of_points = 2 * joint_dimension + 1;

    Filter::SigmaPoints X(state.CohesiveStatesDimension(), number_of_points);

    double epsilon = 1.e-6;

    /*
    // this version is not very useful to debug but sufficient to make sure that
    // everything works
    // iterate over all possible offsets
    for (int offset = 0; offset < max_offset; ++offset)
    {
        filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, offset, X);

        // check if the first column is actually the mean
        EXPECT_TRUE(X.col(0).isApprox(state.state_a_, epsilon));

        // the segment before the offset must be equal to the mean
        for (int i = 1; i < offset + 1; ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(state.state_a_, epsilon));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(state.state_a_, epsilon));
        }

        // the segment starting at the offset contain the distinct sigma points
        for (int i = offset + 1; i < offset + 1 + state.CohesiveStatesDimension(); ++i)
        {
            EXPECT_FALSE(X.col(i).isApprox(state.state_a_, epsilon));
            EXPECT_FALSE(X.col(joint_dimension + i).isApprox(state.state_a_, epsilon));
        }

        // the segment after the offset must be equal to the mean
        for (int i = offset + 1 + state.CohesiveStatesDimension(); i <= joint_dimension; ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(state.state_a_, epsilon));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(state.state_a_, epsilon));
        }
    }
    */

    // for offset = 0
    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 0, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state.state_a_, epsilon));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(2 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(3 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 0).isApprox(state.state_a_, epsilon));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(5 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(6 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 0).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 6 + 0).isApprox(state.state_a_, epsilon));

    //  for offset = 1
    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 1, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state.state_a_, epsilon));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state.state_a_, epsilon));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(2 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(3 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 1).isApprox(state.state_a_, epsilon));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(5 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 1).isApprox(state.state_a_, epsilon));

    //  for offset = 2
    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 2, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state.state_a_, epsilon));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(2).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(state.state_a_, epsilon));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(2 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(3 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 2).isApprox(state.state_a_, epsilon));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 2).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 2).isApprox(state.state_a_, epsilon));

    //  for offset = 3
    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 3, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state.state_a_, epsilon));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(2).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(3).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(state.state_a_, epsilon));
    EXPECT_TRUE(X.col(joint_dimension + 3).isApprox(state.state_a_, epsilon));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 3).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(2 + 3).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(3 + 3).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 3).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 3).isApprox(state.state_a_, epsilon));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 3).isApprox(state.state_a_, epsilon));
}

TEST_F(PartitionedUnscentedTransformTest, partitionUT)
{
    size_t joint_dimension = state.CohesiveStatesDimension()
                              + state.FactorizedStateDimension()
                              + 1;

    size_t number_of_points = 2 * joint_dimension + 1;

    Filter::SigmaPoints Xa(3, number_of_points);

    filter.ComputeSigmaPoints(state.state_a_, state.cov_aa_, 3, Xa);

    std::cout << Xa << std::endl;
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
