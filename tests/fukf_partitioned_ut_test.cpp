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


/**
 * @brief PartitionedUnscentedTransformTest fixture
 */
class PartitionedUnscentedTransformTest:
        public testing::Test
{
public:
    typedef Eigen::Matrix<double, 3, 1> State;

    typedef ProcessModelDummy<State>     ProcessModel;
    typedef ObservationModelDummy<State> ObservationModel;

    typedef ff::FactorizedUnscentedKalmanFilter<
                    ProcessModel,
                    ProcessModel,
                    ObservationModel > Filter;

    PartitionedUnscentedTransformTest():
        filter(Filter(boost::make_shared<ProcessModel>(),
                      boost::make_shared<ProcessModel>(),
                      boost::make_shared<ObservationModel>()))
    {

    }

    virtual void SetUp()
    {
        SetRandom(mu_a,  cov_aa,  Qa,  1.342);
        SetRandom(mu_b,  cov_bb,  Qb,  2.357);
        SetRandom(mu_b2, cov_bb2, Qb2, 5.234);
        SetRandom(mu_y,  cov_yy,  R,   3.523);
    }

    template <typename Mean, typename Covariance, typename NoiseCovaiance>
    void SetRandom(Mean& mu,
                   Covariance& cov,
                   NoiseCovaiance& noise_cov,
                   double sigma)
    {
        mu = Mean::Random();
        cov = Covariance::Random();
        cov = cov * cov.transpose();
        noise_cov = NoiseCovaiance::Identity() * sigma;
    }

    Filter::SigmaPoints AugmentPartitionedSigmaPoints(
            const std::vector<Filter::SigmaPoints>& sigma_point_list)
    {
        size_t dim = 0;
        for (auto& sigma_points: sigma_point_list)
        {
            dim += sigma_points.rows();
        }
        size_t number_of_points = sigma_point_list[0].cols();
        Filter::SigmaPoints joint_partitions_X =
                Filter::SigmaPoints::Zero(dim, number_of_points);

        size_t dim_offset = 0;
        for (auto& sigma_points: sigma_point_list)
        {
            joint_partitions_X.block(dim_offset,
                                     0,
                                     sigma_points.rows(),
                                     number_of_points) = sigma_points;

            dim_offset += sigma_points.rows();
        }

        return joint_partitions_X;
    }

    size_t AugmentedDimension(const std::vector<Eigen::MatrixXd>& matrices)
    {
        size_t dim = 0;
        for (Eigen::MatrixXd matrix : matrices)
        {
            dim += matrix.rows();
        }

        return dim;
    }

    Eigen::MatrixXd AugmentedCovariance(
            const std::vector<Eigen::MatrixXd>& covariances)
    {
        size_t dim = AugmentedDimension(covariances);
        Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(dim, dim);

        size_t offset = 0;
        for (Eigen::MatrixXd covariance : covariances)
        {
            augmented_cov.block(offset,
                                offset,
                                covariance.rows(),
                                covariance.cols()) = covariance;
            offset += covariance.rows();
        }

        return augmented_cov;
    }

    Eigen::MatrixXd AugmentedVector(
            const std::vector<Eigen::MatrixXd>& vectors)
    {
        size_t dim = AugmentedDimension(vectors);
        Eigen::MatrixXd augmented_vector = Eigen::MatrixXd::Zero(dim, dim);

        size_t offset = 0;
        for (Eigen::MatrixXd vector : vectors)
        {
            augmented_vector.block(offset,
                                   0,
                                   vector.rows(),
                                   vector.cols()) = vector;
            offset += vector.rows();
        }

        return augmented_vector;
    }

protected:
    Filter::StateDistribution state;
    Filter filter;
    State mu_a;
    State mu_b;
    State mu_b2;
    Filter::StateDistribution::Y mu_y;

    Filter::StateDistribution::Cov_aa cov_aa;
    Filter::StateDistribution::Cov_bb cov_bb;
    Filter::StateDistribution::Cov_bb cov_bb2;
    Filter::StateDistribution::Cov_yy cov_yy;

    Filter::StateDistribution::Cov_aa Qa;
    Filter::StateDistribution::Cov_bb Qb;
    Filter::StateDistribution::Cov_bb Qb2;
    Filter::StateDistribution::Cov_yy R;
};

const double EPSILON = 1.0e-12;

/**
 * Test the offset behavior
 */
TEST_F(PartitionedUnscentedTransformTest, firstUTColumn)
{
    size_t number_of_points = 2 * state.a_dimension()*5 + 1;

    Filter::SigmaPoints X(state.a_dimension(), number_of_points);
    filter.ComputeSigmaPoints(mu_a, cov_aa, 0, X);

    EXPECT_TRUE(X.col(0).isApprox(mu_a));
}

TEST_F(PartitionedUnscentedTransformTest, partitionUT)
{
    Filter::SigmaPoints augmented_X;
    Filter::SigmaPoints augmented_partitioned_X;

    // compute the sigma point partitions X = [Xa  XQa  Xb  XQb  XR]
    std::vector<Filter::SigmaPoints> X_partitions;

    filter.ComputeSigmaPointPartitions(
        { {mu_a, cov_aa},
          {State::Zero(), Qa},
          {mu_b, cov_bb},
          {State::Zero(), Qb},
          {Filter::StateDistribution::Y::Zero(), R} },
        X_partitions);

    augmented_partitioned_X = AugmentPartitionedSigmaPoints(X_partitions);

    // compute the sigma point from augmented covariance and mean
    size_t dim = AugmentedDimension({cov_aa, Qa, cov_bb, Qb, R});
    size_t number_of_points = 2 * dim + 1;
    Eigen::MatrixXd augmented_mean = AugmentedVector({ mu_a,
                                                       State::Zero(),
                                                       mu_b,
                                                       State::Zero(),
                                                       State::Zero() });
    Eigen::MatrixXd augmented_cov = AugmentedCovariance({ cov_aa,
                                                          Qa,
                                                          cov_bb,
                                                          Qb,
                                                          R });
    augmented_X = Eigen::MatrixXd::Zero(dim, number_of_points);
    filter.ComputeSigmaPoints(augmented_mean, augmented_cov, 0, augmented_X);

    // verify identity
    EXPECT_TRUE(augmented_partitioned_X.isApprox(augmented_X, EPSILON));

}

TEST_F(PartitionedUnscentedTransformTest, partitionCovariance)
{
    Filter::SigmaPoints Xa, Xb, Xy;

    Eigen::MatrixXd S_aa, S_ab, S_ay,
                          S_bb, S_by,
                                S_yy;

    std::vector<Filter::SigmaPoints> X_partitions;

    filter.ComputeSigmaPointPartitions({ {mu_a, cov_aa},
                                         {mu_b, cov_bb},
                                         {mu_y, cov_yy} },
                                       X_partitions);

    Xa = X_partitions[0];
    Xb = X_partitions[1];
    Xy = X_partitions[2];

    size_t dim = Xa.rows() + Xb.rows() + Xy.rows();

    S_aa = Xa * Xa.transpose();
    S_ab = Xa * Xb.transpose();
    S_ay = Xa * Xy.transpose();
    S_bb = Xb * Xb.transpose();
    S_by = Xb * Xy.transpose();
    S_yy = Xy * Xy.transpose();

    Eigen::MatrixXd joint_partitioned_cov = Eigen::MatrixXd::Zero(dim, dim);
    size_t offset_i = 0;
    size_t offset_j = 0;
    joint_partitioned_cov.block(0, offset_j, Xa.rows(), Xa.rows()) = S_aa;

    offset_j += Xa.rows();

    joint_partitioned_cov.block(
                offset_i, offset_j, Xa.rows(), Xb.rows()) = S_ab;
    joint_partitioned_cov.transpose().block(
                offset_i, offset_j, Xa.rows(), Xb.rows()) = S_ab;

    offset_j += Xb.rows();

    joint_partitioned_cov.block(
                offset_i, offset_j, Xa.rows(), Xy.rows()) = S_ay;
    joint_partitioned_cov.transpose().block(
                offset_i, offset_j, Xa.rows(), Xy.rows()) = S_ay;

    offset_i = Xa.rows();
    offset_j = Xa.rows();

    joint_partitioned_cov.block(
                offset_i, offset_j, Xb.rows(), Xb.rows()) = S_bb;

    offset_j += Xb.rows();
    joint_partitioned_cov.block(
                offset_i, offset_j, Xb.rows(), Xy.rows()) = S_by;
    joint_partitioned_cov.transpose().block(
                offset_i, offset_j, Xb.rows(), Xy.rows()) = S_by;

    offset_i = Xa.rows() + Xb.rows();
    offset_j = Xa.rows() + Xb.rows();

    joint_partitioned_cov.block(
                offset_i, offset_j, Xy.rows(), Xy.rows()) = S_yy;


    // compute joint covariance
    Eigen::MatrixXd joint_X = AugmentPartitionedSigmaPoints({Xa, Xb, Xy});
    Eigen::MatrixXd joint_cov = joint_X * joint_X.transpose();

    EXPECT_TRUE(joint_partitioned_cov.isApprox(joint_cov, EPSILON));
}


/**
 * Test the offset behavior
 */
TEST_F(PartitionedUnscentedTransformTest, partitionOffset)
{
    size_t max_offset = 3;
    size_t joint_dimension = state.a_dimension() + max_offset;
    size_t number_of_points = 2 * joint_dimension + 1;

    Filter::SigmaPoints X(state.a_dimension(), number_of_points);

    /*
    // this version is not very useful to debug but sufficient to make sure that
    // everything works
    // iterate over all possible offsets
    for (int offset = 0; offset < max_offset; ++offset)
    {
        filter.ComputeSigmaPoints(mu_a, cov_aa, offset, X);

        // check if the first column is actually the mean
        EXPECT_TRUE(X.col(0).isApprox(mu_a, EPSILON));

        // the segment before the offset must be equal to the mean
        for (int i = 1; i < offset + 1; ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(mu_a, EPSILON));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(mu_a, EPSILON));
        }

        // the segment starting at the offset contain the distinct sigma points
        for (int i = offset + 1;
             i < offset + 1 + state.a_dimension();
             ++i)
        {
            EXPECT_FALSE(X.col(i).isApprox(mu_a, EPSILON));
            EXPECT_FALSE(X.col(joint_dimension + i).isApprox(mu_a, EPSILON));
        }

        // the segment after the offset must be equal to the mean
        for (int i = offset + 1 + state.a_dimension();
             i <= joint_dimension;
             ++i)
        {
            EXPECT_TRUE(X.col(i).isApprox(mu_a, EPSILON));
            EXPECT_TRUE(X.col(joint_dimension + i).isApprox(mu_a, EPSILON));
        }
    }
    */

    // for offset = 0
    filter.ComputeSigmaPoints(mu_a, cov_aa, 0, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(mu_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 0).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 0).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 0).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 0).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 0).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 0).isApprox(mu_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 0).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(5 + 0).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(6 + 0).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 0).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 0).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 6 + 0).isApprox(mu_a, EPSILON));

    //  for offset = 1
    filter.ComputeSigmaPoints(mu_a, cov_aa, 1, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(mu_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(mu_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 1).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 1).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 1).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 1).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 1).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 1).isApprox(mu_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(5 + 1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 5 + 1).isApprox(mu_a, EPSILON));

    //  for offset = 2
    filter.ComputeSigmaPoints(mu_a, cov_aa, 2, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(mu_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(2).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(mu_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 2).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 2).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 2).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 2).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 2).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 2).isApprox(mu_a, EPSILON));
    // the segment after the offset must be equal to the mean
    EXPECT_TRUE(X.col(4 + 2).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 4 + 2).isApprox(mu_a, EPSILON));

    //  for offset = 3
    filter.ComputeSigmaPoints(mu_a, cov_aa, 3, X);
    // check if the first column is actually the mean
    EXPECT_TRUE(X.col(0).isApprox(mu_a, EPSILON));
    // the segment before the offset must be equal to the mean
    EXPECT_TRUE(X.col(1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(2).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(3).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 1).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 2).isApprox(mu_a, EPSILON));
    EXPECT_TRUE(X.col(joint_dimension + 3).isApprox(mu_a, EPSILON));
    // the segment starting at the offset contain the distinct sigma points
    EXPECT_FALSE(X.col(1 + 3).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(2 + 3).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(3 + 3).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 1 + 3).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 2 + 3).isApprox(mu_a, EPSILON));
    EXPECT_FALSE(X.col(joint_dimension + 3 + 3).isApprox(mu_a, EPSILON));
}
