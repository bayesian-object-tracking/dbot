/*
 * This is part of the FL library, a C++ Bayesian filtering library
 * (https://github.com/filtering-library)
 *
 * Copyright (c) 2014 Jan Issac (jan.issac@gmail.com)
 * Copyright (c) 2014 Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 * Max-Planck Institute for Intelligent Systems, AMD Lab
 * University of Southern California, CLMC Lab
 *
 * This Source Code Form is subject to the terms of the MIT License (MIT).
 * A copy of the license can be found in the LICENSE file distributed with this
 * source code.
 */

/**
 * \date January 2015
 * \author Jan Issac (jan.issac@gmail.com)
 */

#include <fstream>
#include <ctime>
#include <memory>
#include <iostream>

#include <std_msgs/Header.h>
#include <ros/ros.h>
#include <ros/package.h>

#include <fl/util/meta.hpp>
#include <fl/util/profiling.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>
#include <fl/filter/gaussian/gaussian_filter_factorized.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/monte_carlo_transform.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

#include <fl/model/observation/joint_observation_model.hpp>
#include <fl/model/process/joint_process_model.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/image_publisher.hpp>

#include "depth_pixel_model.hpp"
#include "squared_feature_policy.hpp"

enum : signed int
{
    StateDimension = 1,
    DepthPixelDimension = 1,
    PixelCount = Eigen::Dynamic
};

using namespace fl;

template <typename Option, int SigmaPoints>
class GfContext
{
public:
    /* ############################## */
    /* # Basic types                # */
    /* ############################## */
    typedef double Scalar;

    // simple 1D state
    typedef Eigen::Matrix<Scalar, StateDimension, 1> CurveState;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    typedef PixelModel<CurveState> SinglePixelModel;

    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef LinearGaussianProcessModel<CurveState> ProcessModel;

    /* ############################## */
    /* # Observation Param Model    # */
    /* ############################## */
    typedef typename Traits<SinglePixelModel>::Param Param;
    typedef LinearGaussianProcessModel<Param> ParamModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef MonteCarloTransform<
                ConstantPointCountPolicy<SigmaPoints>
            > PointTransform;

    //typedef IdentityFeaturePolicy<> FeaturePolicy;
//    typedef SquaredFeaturePolicy<> FeaturePolicy;
    typedef SigmoidFeaturePolicy<> FeaturePolicy;

    typedef GaussianFilter<
                ProcessModel,
                Join<MultipleOf<Adaptive<SinglePixelModel, ParamModel>, PixelCount>>,
                PointTransform,
                FeaturePolicy,
                Option
            > FilterAlgo;

    typedef FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::State State;
    typedef typename Filter::Input Input;
    typedef typename Filter::Obsrv Obsrv;
    typedef typename Filter::StateDistribution StateDistribution;
    typedef typename Traits<FilterAlgo>::FeatureMapping FeatureMapping;

    GfContext(ros::NodeHandle& nh)
        : filter_(
              create_filter(nh,
                  create_process_model(nh),
                  create_param_model(nh),
                  create_pixel_model(nh))),
          state_distr_(filter_.create_state_distribution()),
          zero_input_(filter_.process_model().input_dimension(), 1),
          y(filter_.obsrv_model().obsrv_dimension(), 1)
    {
        zero_input_.setZero();
    }

    void filter(const Obsrv& y)
    {
        filter_.predict(1.0, zero_input_, state_distr_, state_distr_);
        filter_.update(y, state_distr_, state_distr_);
    }

    FilterAlgo& algo() { return filter_; }

public:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    ProcessModel create_process_model(ros::NodeHandle& nh)
    {
        double A_a;
        double v_sigma_a;
        ri::ReadParameter("A_a", A_a, nh);
        ri::ReadParameter("v_sigma_a", v_sigma_a, nh);

        auto model = ProcessModel();

        // set noise covariance
        model.A(model.A() * A_a);
        model.covariance(model.covariance() * std::pow(v_sigma_a, 2));

        return model;
    }

    ParamModel create_param_model(ros::NodeHandle& nh)
    {
        double A_b;
        double v_sigma_b;
        ri::ReadParameter("A_b", A_b, nh);
        ri::ReadParameter("v_sigma_b", v_sigma_b, nh);

        auto model = ParamModel();

        // set dynamics matrix and noise covariance
        model.A(model.A() * A_b);
        model.covariance(model.covariance() * std::pow(v_sigma_b, 2));

        return model;
    }

    SinglePixelModel create_pixel_model(ros::NodeHandle& nh)
    {
        double w_sigma;
        double w_sigma_b;
        ri::ReadParameter("w_sigma", w_sigma, nh);

        auto model = SinglePixelModel();

        // set param initial value and noise covariance
        model.covariance(model.covariance() * std::pow(w_sigma, 2));

        return model;
    }

    FilterAlgo create_filter(ros::NodeHandle& nh,
                             ProcessModel process_model,
                             ParamModel param_model,
                             SinglePixelModel pixel_model)
    {
        int pixel_count;
        ri::ReadParameter("pixel_count", pixel_count, nh);

        auto filter = FilterAlgo(process_model,
                                 param_model,
                                 pixel_model,
                                 PointTransform(),
                                 FeatureMapping(),
                                 pixel_count);

        return filter;
    }        

public:
    FilterAlgo filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;

    int pixel_count;
    int error_pixel;
    double error_pixel_depth;
    bool factorize_;
};

