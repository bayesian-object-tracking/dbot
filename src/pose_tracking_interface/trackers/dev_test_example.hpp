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
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/monte_carlo_transform.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/image_publisher.hpp>

#include "dev_test_tracker/pixel_observation_model.hpp"
#include "dev_test_tracker/depth_observation_model.hpp"


static constexpr int StateDimension = 1;
static constexpr int ObsrvDimension = -1;
static constexpr int ParametersDimension = ObsrvDimension;

class DevTestExample
{
public:
    /* ############################## */
    /* # Basic types                # */
    /* ############################## */
    typedef double Scalar;
    typedef Eigen::Matrix<
                Scalar,
                fl::JoinSizes<StateDimension, ParametersDimension>::Size,
                1
            > State;

    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef fl::LinearGaussianProcessModel<State> ProcessModel;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    /* Pixel Model */
//    typedef Eigen::Matrix<double, ObsrvDimension, 1> Observation;
//    typedef fl::LinearGaussianObservationModel<Observation, State> ObservationModel;
//    typedef typename fl::Traits<ObservationModel>::SecondMoment Cov;
    typedef fl::PixelObservationModel<Scalar> PixelObsrvModel;
    typedef fl::ExperimentalObservationModel<
                    PixelObsrvModel,
                    State,
                    ObsrvDimension
                > ObservationModel;
    typedef typename fl::Traits<ObservationModel>::Observation Observation;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::MonteCarloTransform<
                //fl::LinearPointCountPolicy<2>
                fl::ConstantPointCountPolicy<2000>
                //fl::MonomialPointCountPolicy<1>
            > PointTransform;

    typedef fl::GaussianFilter<
                ProcessModel,
                ObservationModel,
                PointTransform
            > FilterAlgo;

    typedef fl::FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::Input Input;
    typedef typename Filter::StateDistribution StateDistribution;

    DevTestExample(ros::NodeHandle& nh)
        : process_model_(create_process_model(nh)),
          obsrv_model_(create_obsrv_model(nh)),
          filter_(create_filter(nh, process_model_, obsrv_model_)),

          state_distr_(filter_->process_model()->state_dimension()),
          zero_input_(filter_->process_model()->input_dimension(), 1),
          y(filter_->observation_model()->observation_dimension(), 1)
    {
        State initial_state(filter_->process_model()->state_dimension(), 1);

        initial_state.setZero();
        zero_input_.setZero();

        state_distr_.mean(initial_state);
        state_distr_.covariance(state_distr_.covariance() * 1.e-12);

        ri::ReadParameter("print_ukf_details", filter_->print_details, nh);
        ri::ReadParameter("factorize", factorize_, nh);
    }

    void filter(const Observation& y)
    {
        filter_->predict(1.0, zero_input_, state_distr_, state_distr_);

        filter_->update(y, state_distr_, state_distr_);

        if (factorize_)
        {
            auto iden = state_distr_.covariance();
            iden.setIdentity();
            state_distr_.covariance(state_distr_.covariance().cwiseProduct(iden));
        }
    }

private:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    /**
     * \return Process model instance
     */
    std::shared_ptr<ProcessModel> create_process_model(ros::NodeHandle& nh)
    {
        double A_b;
        double v_a_sigma;
        double v_b_sigma;
        int pixels;
        int state_dim;

        ri::ReadParameter("A_b", A_b, nh);
        ri::ReadParameter("v_a_sigma", v_a_sigma, nh);
        ri::ReadParameter("v_b_sigma", v_b_sigma, nh);
        ri::ReadParameter("pixels", pixels, nh);
        state_dim = 1 + pixels;

        typedef typename fl::Traits<ProcessModel>::SecondMoment Cov;

        auto model = std::make_shared<ProcessModel>(
                        Cov::Identity(state_dim, state_dim),
                        state_dim);

        auto A = model->A();
        auto cov = model->covariance();

        assert(cov.rows() == state_dim);
        assert(cov.cols() == state_dim);
        assert(A.rows() == state_dim);
        assert(A.cols() == state_dim);

        cov(0, 0) = v_a_sigma * v_a_sigma;
        cov.block(1, 1, pixels, pixels) =
                cov.block(1, 1, pixels, pixels) * v_b_sigma * v_b_sigma;

        A.block(1, 1, pixels, pixels) = A.block(1, 1, pixels, pixels) * A_b;

        model->covariance(cov);
        model->A(A);

        return model;
    }

    /**
     * \return Observation model instance
     */
    std::shared_ptr<ObservationModel> create_obsrv_model(ros::NodeHandle& nh)
    {
        double w_sigma;
        double max_w_sigma;
        int sigma_function;
        ri::ReadParameter("w_sigma", w_sigma, nh);
        ri::ReadParameter("max_w_sigma", max_w_sigma, nh);
        ri::ReadParameter("pixels", pixels, nh);
        ri::ReadParameter("sigma_function", sigma_function, nh);

//        return std::make_shared<ObservationModel>
//        (
//            Cov::Identity(pixels, pixels) * w_sigma * w_sigma,
//            pixels,
//            StateDimension
//        );

        return std::make_shared<ObservationModel>
        (
            std::make_shared<PixelObsrvModel>(
                        w_sigma * w_sigma, max_w_sigma, sigma_function),
            pixels + 1,
            pixels
        );
    }

    /**
     * \return Filter instance
     */
    Filter::Ptr create_filter(
            ros::NodeHandle& nh,
            const std::shared_ptr<ProcessModel>& process_model,
            const std::shared_ptr<ObservationModel>& obsrv_model)
    {
        return std::make_shared<FilterAlgo>
        (
            process_model,
            obsrv_model,
            std::make_shared<PointTransform>()
        );
    }

public:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObservationModel> obsrv_model_;
    Filter::Ptr filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Observation y;

    int pixels;
    int error_pixel;
    double error_pixel_depth;
    bool factorize_;
};
