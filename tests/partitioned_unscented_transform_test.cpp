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
        state_a = State::Random();
        cov_aa = Filter::StateDistribution::CovAA::Random();
        cov_aa = cov_aa * cov_aa.transpose();
        Qa = Filter::StateDistribution::CovAA::Identity() * 5;

        state_b = State::Random();
        cov_bb = Filter::StateDistribution::CovBB::Random();
        cov_bb = cov_bb * cov_bb.transpose();
        Qb = Filter::StateDistribution::CovAA::Identity() * 3;

        state_b2 = State::Random();
        cov_bb2 = Filter::StateDistribution::CovBB::Random();
        cov_bb2 = cov_bb2 * cov_bb2.transpose();
        Qb2 = Filter::StateDistribution::CovAA::Identity() * 3;

        y = Filter::StateDistribution::Y::Random();
        cov_yy = Filter::StateDistribution::CovYY::Random();
        cov_yy = cov_yy * cov_yy.transpose();
        R = Filter::StateDistribution::CovYY::Identity() * 2;
    }

    virtual size_t JointDimension()
    {
        return state.CohesiveStatesDimension() * 2
                + state.FactorizedStateDimension() * 2
                + 1;
    }

    virtual size_t JointDimensionNoNoise()
    {
        return state.CohesiveStatesDimension()
                + state.FactorizedStateDimension()
                + 1;
    }

    void ComputeSigmaPointPartitions(Filter::SigmaPoints& Xa,
                                     Filter::SigmaPoints& XQa,
                                     Filter::SigmaPoints& Xb,
                                     Filter::SigmaPoints& XQb,
                                     Filter::SigmaPoints& XR)
    {
        size_t number_of_points = 2 * JointDimension() + 1;
        size_t offset = 0;

        Xa = Filter::SigmaPoints(state.CohesiveStatesDimension(), number_of_points);
        XQa = Filter::SigmaPoints(state.CohesiveStatesDimension(), number_of_points);

        Xb = Filter::SigmaPoints(state.FactorizedStateDimension(), number_of_points);
        XQb = Filter::SigmaPoints(state.FactorizedStateDimension(), number_of_points);
        XR = Filter::SigmaPoints(1, number_of_points);

        filter.ComputeSigmaPoints(state_a, cov_aa, offset, Xa);
        offset += state.CohesiveStatesDimension();

        filter.ComputeSigmaPoints<State>(State::Zero(), Qa, offset, XQa);
        offset += state.CohesiveStatesDimension();

        filter.ComputeSigmaPoints(state_b, cov_bb, offset, Xb);
        offset += state.FactorizedStateDimension();

        filter.ComputeSigmaPoints<State>(State::Zero(), Qb, offset, XQb);
        offset += state.FactorizedStateDimension();

        filter.ComputeSigmaPoints<Filter::StateDistribution::Y>(
                    Filter::StateDistribution::Y::Zero(), R, offset, XR);

    }

    void ComputeSigmaPointPartitionsNoNoise(Filter::SigmaPoints& Xa,
                                            Filter::SigmaPoints& Xb,
                                            Filter::SigmaPoints& XR)
    {
        size_t number_of_points = 2 * JointDimensionNoNoise() + 1;
        size_t offset = 0;

        Xa = Filter::SigmaPoints(state.CohesiveStatesDimension(), number_of_points);
        Xb = Filter::SigmaPoints(state.FactorizedStateDimension(), number_of_points);
        XR = Filter::SigmaPoints(1, number_of_points);

        filter.ComputeSigmaPoints(state_a, cov_aa, offset, Xa);
        offset += state.CohesiveStatesDimension();

        filter.ComputeSigmaPoints(state_b, cov_bb, offset, Xb);
        offset += state.FactorizedStateDimension();

        filter.ComputeSigmaPoints<Filter::StateDistribution::Y>(
                    Filter::StateDistribution::Y::Zero(), R, offset, XR);

    }

    Eigen::MatrixXd JointPartitionedSigmaPoints(
            Filter::SigmaPoints& Xa,
            Filter::SigmaPoints& XQa,
            Filter::SigmaPoints& Xb,
            Filter::SigmaPoints& XQb,
            Filter::SigmaPoints& XR)
    {
        size_t number_of_points = 2 * JointDimension() + 1;

        Eigen::MatrixXd joint_partitions_X = Eigen::MatrixXd::Zero(JointDimension(), number_of_points);

        size_t dim_offset = 0;
        joint_partitions_X.block(dim_offset, 0, Xa.rows(), number_of_points) = Xa;
        dim_offset += Xa.rows();
        joint_partitions_X.block(dim_offset, 0, XQa.rows(), number_of_points) = XQa;
        dim_offset += XQa.rows();
        joint_partitions_X.block(dim_offset, 0, Xb.rows(), number_of_points) = Xb;
        dim_offset += Xb.rows();
        joint_partitions_X.block(dim_offset, 0, XQb.rows(), number_of_points) = XQb;
        dim_offset += XQb.rows();
        joint_partitions_X.block(dim_offset, 0, XR.rows(), number_of_points) = XR;

        return joint_partitions_X;
    }

    Eigen::MatrixXd JointPartitionedSigmaPointsNoNoise(
            Filter::SigmaPoints& Xa,
            Filter::SigmaPoints& Xb,
            Filter::SigmaPoints& XR)
    {
        size_t number_of_points = 2 * JointDimensionNoNoise() + 1;

        Eigen::MatrixXd joint_partitions_X = Eigen::MatrixXd::Zero(JointDimensionNoNoise(), number_of_points);

        size_t dim_offset = 0;
        joint_partitions_X.block(dim_offset, 0, Xa.rows(), number_of_points) = Xa;
        dim_offset += Xa.rows();
        joint_partitions_X.block(dim_offset, 0, Xb.rows(), number_of_points) = Xb;
        dim_offset += Xb.rows();
        joint_partitions_X.block(dim_offset, 0, XR.rows(), number_of_points) = XR;

        return joint_partitions_X;
    }

    Eigen::MatrixXd JointSigmaPoints()
    {
        size_t number_of_points = 2 * JointDimension() + 1;
        size_t offset = 0;


        size_t dim_offset = 0;

        Eigen::MatrixXd joint_mean = Eigen::MatrixXd::Zero(JointDimension(), 1);
        Eigen::MatrixXd joint_cov = Eigen::MatrixXd::Zero(JointDimension(), JointDimension());

        joint_mean.block(dim_offset, 0, state_a.rows(), 1) = state_a;
        joint_cov.block(dim_offset, dim_offset, cov_aa.rows(), cov_aa.cols()) = cov_aa;
        dim_offset += cov_aa.rows();

        joint_mean.block(dim_offset, 0, state_a.rows(), 1) = State::Zero();
        joint_cov.block(dim_offset, dim_offset, Qa.rows(), Qa.cols()) = Qa;
        dim_offset += Qa.rows();

        joint_mean.block(dim_offset, 0, state_b.rows(), 1) = state_b;
        joint_cov.block(dim_offset, dim_offset, cov_bb.rows(), cov_bb.cols()) = cov_bb;
        dim_offset += cov_bb.rows();

        joint_mean.block(dim_offset, 0, state_b.rows(), 1) = State::Zero();
        joint_cov.block(dim_offset, dim_offset, Qb.rows(), Qb.cols()) = Qb;
        dim_offset += Qb.rows();

        joint_mean.block(dim_offset, 0, y.rows(), 1) = Filter::StateDistribution::Y::Zero();
        joint_cov.block(dim_offset, dim_offset, R.rows(), R.cols()) = R;

        Eigen::MatrixXd joint_X = Eigen::MatrixXd::Zero(JointDimension(), number_of_points);

        filter.ComputeSigmaPoints(joint_mean, joint_cov, 0, joint_X);

        return joint_X;
    }


    Eigen::MatrixXd JointSigmaPointsNoNoise()
    {
        size_t number_of_points = 2 * JointDimensionNoNoise() + 1;
        size_t offset = 0;


        size_t dim_offset = 0;

        Eigen::MatrixXd joint_mean = Eigen::MatrixXd::Zero(JointDimensionNoNoise(), 1);
        Eigen::MatrixXd joint_cov = Eigen::MatrixXd::Zero(JointDimensionNoNoise(), JointDimensionNoNoise());

        joint_mean.block(dim_offset, 0, state_a.rows(), 1) = state_a;
        joint_cov.block(dim_offset, dim_offset, cov_aa.rows(), cov_aa.cols()) = cov_aa;
        dim_offset += cov_aa.rows();

        joint_mean.block(dim_offset, 0, state_b.rows(), 1) = state_b;
        joint_cov.block(dim_offset, dim_offset, cov_bb.rows(), cov_bb.cols()) = cov_bb;
        dim_offset += cov_bb.rows();

        joint_mean.block(dim_offset, 0, y.rows(), 1) = Filter::StateDistribution::Y::Zero();
        joint_cov.block(dim_offset, dim_offset, R.rows(), R.cols()) = R;

        Eigen::MatrixXd joint_X = Eigen::MatrixXd::Zero(JointDimensionNoNoise(), number_of_points);

        filter.ComputeSigmaPoints(joint_mean, joint_cov, 0, joint_X);

        return joint_X;
    }

protected:
    Filter::StateDistribution state;
    Filter filter;
    State state_a;
    State state_b;
    State state_b2;
    Filter::StateDistribution::Y y;

    Filter::StateDistribution::CovAA cov_aa;
    Filter::StateDistribution::CovBB cov_bb;
    Filter::StateDistribution::CovBB cov_bb2;
    Filter::StateDistribution::CovYY cov_yy;

    Filter::StateDistribution::CovAA Qa;
    Filter::StateDistribution::CovBB Qb;
    Filter::StateDistribution::CovBB Qb2;
    Filter::StateDistribution::CovYY R;
};

const double EPSILON = 1.0e-12;

/**
 * Test the offset behavior
 */
TEST_F(PartitionedUnscentedTransformTest, firstUTColumn)
{
    size_t number_of_points = 2 * JointDimension() + 1;

    Filter::SigmaPoints X(state.CohesiveStatesDimension(), number_of_points);
    filter.ComputeSigmaPoints(state_a, cov_aa, 0, X);

    EXPECT_TRUE(X.col(0).isApprox(state_a));
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

    /*
    // this version is not very useful to debug but sufficient to make sure that
    // everything works
    // iterate over all possible offsets
    for (int offset = 0; offset < max_offset; ++offset)
    {
        filter.ComputeSigmaPoints(state_a, cov_aa, offset, X);

        // check if the first column is actually the mean
        EXPECT_TRUE(X.col(0).isApprox(state_a, EPSILON));

        // the segment before the offset must be equal to the mean
        for (int i = 1; i < offset + 1; ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(state_a, EPSILON));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(state_a, EPSILON));
        }

        // the segment starting at the offset contain the distinct sigma points
        for (int i = offset + 1; i < offset + 1 + state.CohesiveStatesDimension(); ++i)
        {
            EXPECT_FALSE(X.col(i).isApprox(state_a, EPSILON));
            EXPECT_FALSE(X.col(joint_dimension + i).isApprox(state_a, EPSILON));
        }

        // the segment after the offset must be equal to the mean
        for (int i = offset + 1 + state.CohesiveStatesDimension(); i <= joint_dimension; ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(state_a, EPSILON));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(state_a, EPSILON));
        }
    }
    */

    // for offset = 0
    filter.ComputeSigmaPoints(state_a, cov_aa, 0, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 0).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 0).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 0).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 0).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 0).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 0).isApprox(state_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 0).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(5 + 0).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(6 + 0).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 0).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 0).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 6 + 0).isApprox(state_a, EPSILON));

    //  for offset = 1
    filter.ComputeSigmaPoints(state_a, cov_aa, 1, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 1).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 1).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 1).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 1).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 1).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 1).isApprox(state_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(5 + 1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 1).isApprox(state_a, EPSILON));

    //  for offset = 2
    filter.ComputeSigmaPoints(state_a, cov_aa, 2, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(2).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(state_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 2).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 2).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 2).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 2).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 2).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 2).isApprox(state_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 2).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 2).isApprox(state_a, EPSILON));

    //  for offset = 3
    filter.ComputeSigmaPoints(state_a, cov_aa, 3, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(state_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(2).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(3).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(state_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 3).isApprox(state_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 3).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 3).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 3).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 3).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 3).isApprox(state_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 3).isApprox(state_a, EPSILON));
}

TEST_F(PartitionedUnscentedTransformTest, partitionUT)
{
    Filter::SigmaPoints Xa;
    Filter::SigmaPoints XQa;
    Filter::SigmaPoints Xb;
    Filter::SigmaPoints XQb;
    Filter::SigmaPoints XR;

    ComputeSigmaPointPartitions(Xa, XQa, Xb, XQb, XR);

    Eigen::MatrixXd joint_partitions_X = JointPartitionedSigmaPoints(Xa, XQa, Xb, XQb, XR);

    Eigen::MatrixXd joint_X = JointSigmaPoints();

    EXPECT_TRUE(joint_partitions_X.isApprox(joint_X, EPSILON));
}

TEST_F(PartitionedUnscentedTransformTest, partitionUTNoNoise)
{
    Filter::SigmaPoints Xa;
    Filter::SigmaPoints Xb;
    Filter::SigmaPoints XR;
    ComputeSigmaPointPartitionsNoNoise(Xa, Xb, XR);
    Eigen::MatrixXd joint_partitions_X = JointPartitionedSigmaPointsNoNoise(Xa, Xb, XR);

    Eigen::MatrixXd joint_X = JointSigmaPointsNoNoise();

    EXPECT_TRUE(joint_partitions_X.isApprox(joint_X, EPSILON));
}

TEST_F(PartitionedUnscentedTransformTest, partitionCovariance)
{
    Filter::SigmaPoints Xa;
    Filter::SigmaPoints Xb;
    Filter::SigmaPoints XR;

    ComputeSigmaPointPartitionsNoNoise(Xa, Xb, XR);

    Eigen::MatrixXd joint_partition_cov = Eigen::MatrixXd::Zero(Xa.rows()+Xb.rows()+XR.rows(), Xa.rows()+Xb.rows()+XR.rows());

    size_t offset_i = 0;
    size_t offset_j = 0;
    joint_partition_cov.block(0, offset_j, Xa.rows(), Xa.rows()) = Xa * Xa.transpose();

    offset_j += Xa.rows();
    joint_partition_cov.block(0, offset_j, Xa.rows(), Xb.rows()) = Xa * Xb.transpose();
    joint_partition_cov.transpose().block(0, offset_j, Xa.rows(), Xb.rows()) = Xa * Xb.transpose();

    offset_j += Xb.rows();
    joint_partition_cov.block(0, offset_j, Xa.rows(), XR.rows()) = Xa * XR.transpose();
    joint_partition_cov.transpose().block(0, offset_j, Xa.rows(), XR.rows()) = Xa * XR.transpose();

    offset_i = Xa.rows();
    offset_j = Xa.rows();
    joint_partition_cov.block(offset_i, offset_j, Xb.rows(), Xb.rows()) = Xb * Xb.transpose();

    offset_j += Xb.rows();
    joint_partition_cov.block(offset_i, offset_j, Xb.rows(), XR.rows()) = Xb * XR.transpose();
    joint_partition_cov.transpose().block(offset_i, offset_j, Xb.rows(), XR.rows()) = Xb * XR.transpose();

    offset_i = Xa.rows() + Xb.rows();
    offset_j = Xa.rows() + Xb.rows();
    joint_partition_cov.block(offset_i, offset_j, XR.rows(), XR.rows()) = XR * XR.transpose();

    Eigen::MatrixXd joint_X = JointSigmaPointsNoNoise();
    Eigen::MatrixXd joint_cov = joint_X * joint_X.transpose();

    EXPECT_TRUE(joint_partition_cov.isApprox(joint_cov, EPSILON));
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}





















