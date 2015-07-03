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

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <fast_filtering/models/process_models/linear_process_model.hpp>
#include <fast_filtering/models/observation_models/linear_observation_model.hpp>
#include <fast_filtering/filters/deterministic/kalman_filter.hpp>

TEST(KalmanFilterTests, init_fixedsize_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State;
    typedef Eigen::Matrix<Scalar, 20, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State> ProcessModel;
    typedef ff::LinearGaussianOservationModel<Observation,
                                              State> ObservationModel;

    typedef ff::KalmanFilter<ProcessModel, ObservationModel> Filter;

    ProcessModel::Operator Q = ProcessModel::Operator::Identity();
    ObservationModel::Operator R = ObservationModel::Operator::Identity();

    Filter::ProcessModelPtr process_model =
            boost::make_shared<ProcessModel>(Q);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R);

    Filter filter(process_model, observation_model);

    Filter::StateDistribution state_dist;

    EXPECT_TRUE(state_dist.Mean().isZero());
    EXPECT_TRUE(state_dist.Covariance().isIdentity());
    filter.Predict(1.0, state_dist, state_dist);
    EXPECT_TRUE(state_dist.Mean().isZero());
    EXPECT_TRUE(state_dist.Covariance().isApprox(2. * Q));

}

TEST(KalmanFilterTests, init_dynamicsize_predict)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    const size_t dim_state = 10;
    const size_t dim_observation = 20;

    typedef ff::LinearGaussianProcessModel<State> ProcessModel;
    typedef ff::LinearGaussianOservationModel<Observation,
                                              State> ObservationModel;

    typedef ff::KalmanFilter<ProcessModel, ObservationModel> Filter;

    ProcessModel::Operator Q =
            ProcessModel::Operator::Identity(dim_state,
                                             dim_state);
    ObservationModel::Operator R =
            ObservationModel::Operator::Identity(dim_observation,
                                                 dim_observation);

    Filter::ProcessModelPtr process_model =
            boost::make_shared<ProcessModel>(Q, dim_state);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R, dim_observation, dim_state);

    Filter filter(process_model, observation_model);

    Filter::StateDistribution state_dist(dim_state);

    EXPECT_TRUE(state_dist.Mean().isZero());
    EXPECT_TRUE(state_dist.Covariance().isIdentity());
    filter.Predict(1.0, state_dist, state_dist);
    EXPECT_TRUE(state_dist.Mean().isZero());
    EXPECT_TRUE(state_dist.Covariance().isApprox(2. * Q));
}


TEST(KalmanFilterTests, fixedsize_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 10, 1> State;
    typedef Eigen::Matrix<Scalar, 10, 1> Observation;

    typedef ff::LinearGaussianProcessModel<State> ProcessModel;
    typedef ff::LinearGaussianOservationModel<Observation,
                                              State> ObservationModel;

    typedef ff::KalmanFilter<ProcessModel, ObservationModel> Filter;

    ProcessModel::Operator Q = ProcessModel::Operator::Random()*1.5;
    Q *= Q.transpose();
    ProcessModel::DynamicsMatrix A = ProcessModel::DynamicsMatrix::Random();

    ObservationModel::Operator R = ObservationModel::Operator::Random();
    R *= R.transpose();
    ObservationModel::SensorMatrix H = ObservationModel::SensorMatrix::Random();

    Filter::ProcessModelPtr process_model =
            boost::make_shared<ProcessModel>(Q);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R);

    process_model->A(A);
    observation_model->H(H);

    Filter filter(process_model, observation_model);

    Filter::StateDistribution state_dist;
    EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());

    for (size_t i = 0; i < 20000; ++i)
    {
        filter.Predict(1.0, state_dist, state_dist);
        EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());
        Observation y = Observation::Random();
        filter.Update(state_dist, y, state_dist);
        EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());
    }
}


TEST(KalmanFilterTests, dynamicsize_predict_update)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> State;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Observation;

    const size_t dim_state = 10;
    const size_t dim_observation = 10;

    typedef ff::LinearGaussianProcessModel<State> ProcessModel;
    typedef ff::LinearGaussianOservationModel<Observation,
                                              State> ObservationModel;

    typedef ff::KalmanFilter<ProcessModel, ObservationModel> Filter;

    ProcessModel::Operator Q = ProcessModel::Operator::Random(dim_state, dim_state)*1.5;
    Q *= Q.transpose();
    ProcessModel::DynamicsMatrix A = ProcessModel::DynamicsMatrix::Random(dim_state, dim_state);

    ObservationModel::Operator R = ObservationModel::Operator::Random(dim_observation, dim_observation);
    R *= R.transpose();
    ObservationModel::SensorMatrix H = ObservationModel::SensorMatrix::Random(dim_observation, dim_state);

    Filter::ProcessModelPtr process_model =
            boost::make_shared<ProcessModel>(Q, dim_state);
    Filter::ObservationModelPtr observation_model =
            boost::make_shared<ObservationModel>(R, dim_observation, dim_state);

    process_model->A(A);
    observation_model->H(H);

    Filter filter(process_model, observation_model);

    Filter::StateDistribution state_dist(dim_state);
    EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());

    for (size_t i = 0; i < 20000; ++i)
    {
        filter.Predict(1.0, state_dist, state_dist);
        EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());
        Observation y = Observation::Random(dim_observation, 1);
        filter.Update(state_dist, y, state_dist);
        EXPECT_TRUE(state_dist.Covariance().ldlt().isPositive());
    }
}










