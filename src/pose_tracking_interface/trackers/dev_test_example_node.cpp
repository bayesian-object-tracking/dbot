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
#include <std_msgs/Float32.h>

#include <fl/util/meta.hpp>
#include <fl/util/profiling.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/random_transform.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/model/observation/linear_observation_model.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/image_publisher.hpp>

#include "dev_test_tracker/pixel_observation_model.hpp"

class DevTestExample
{
public:
    static constexpr int StateDimension = 1;
    static constexpr int ObsrvDimension = 1;
    static constexpr int ParamDimension = ObsrvDimension;

    /* ############################## */
    /* # Basic types                # */
    /* ############################## */
    typedef double Scalar;
    typedef Eigen::Matrix<
                Scalar,
                //fl::JoinSizes<StateDimension, ParamDimension>::Size,
                1,
                1
            > State;

    typedef Eigen::Matrix<double, ObsrvDimension, 1> Obsrv;

    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef fl::LinearGaussianProcessModel<State> ProcessModel;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    /* Pixel Model */
    //typedef fl::PixelObservationModel<Scalar> PixelObsrvModel;
    typedef fl::LinearGaussianObservationModel<Obsrv, State> PixelObsrvModel;

    /* Camera Model(Pixel Model) */
    typedef fl::FactorizedIIDObservationModel<PixelObsrvModel> ObsrvModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObsrvModel,
                fl::RandomTransform
            > FilterAlgo;

    typedef fl::FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::Input Input;
    //typedef typename Filter::Observation Obsrv;
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
    }

    void filter(const Obsrv& y)
    {
        filter_->predict(1.0, zero_input_, state_distr_, state_distr_);
        filter_->update(y, state_distr_, state_distr_);
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
        double A_factor;
        double v_sigma;

        ri::ReadParameter("A_factor", A_factor, nh);
        ri::ReadParameter("v_sigma", v_sigma, nh);

        typedef typename fl::Traits<ProcessModel>::SecondMoment Cov;

        auto model = std::make_shared<ProcessModel>(Cov::Identity() * v_sigma);
        model->A(model->A() * A_factor);

        return model;
    }

    /**
     * \return Observation model instance
     */
    std::shared_ptr<ObsrvModel> create_obsrv_model(ros::NodeHandle& nh)
    {
        double w_sigma;
        ri::ReadParameter("w_sigma", w_sigma, nh);
        //ri::ReadParameter("pixels", pixels, nh);
        pixels = ParamDimension;

        typedef typename fl::Traits<PixelObsrvModel>::SecondMoment Cov;

        return std::make_shared<ObsrvModel>
        (
            std::make_shared<PixelObsrvModel>(
                Cov::Identity() * w_sigma * w_sigma,
                ObsrvDimension,
                StateDimension),
            pixels
        );
    }

    /**
     * \return Filter instance
     */
    Filter::Ptr create_filter(
            ros::NodeHandle& nh,
            const std::shared_ptr<ProcessModel>& process_model,
            const std::shared_ptr<ObsrvModel>& obsrv_model)
    {
        return std::make_shared<FilterAlgo>
        (
            process_model,
            obsrv_model,
            std::make_shared<fl::RandomTransform>()
        );
    }

public:
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
    Filter::Ptr filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;

    int pixels;
    int error_pixel;
    double error_pixel_depth;
};


int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "dev_test_example");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    DevTestExample tracker(nh);
    DevTestExample::Obsrv y(DevTestExample::ObsrvDimension,1);

    /* ############################## */
    /* # Images                     # */
    /* ############################## */
    fl::ImagePublisher ip(nh);

    Eigen::MatrixXd image_vector(tracker.pixels, 1);

    std::cout << ">> initial state " << std::endl;
    std::cout << tracker.state_distr_.mean().transpose() << std::endl;

    std::cout << ">> setup: " << std::endl;    
    std::cout << ">> state dimension: "
              << tracker.process_model_->state_dimension() << std::endl;
    std::cout << ">> noise dimension: "
              << tracker.process_model_->noise_dimension() << std::endl;
    std::cout << ">> obsrv dimension: "
              << tracker.obsrv_model_->observation_dimension() << std::endl;
    std::cout << ">> obsrv noise dimension: "
              << tracker.obsrv_model_->noise_dimension() << std::endl;
    std::cout << ">> obsrv state dimension: "
              << tracker.obsrv_model_->state_dimension() << std::endl;

    std::cout << ">> sigma point number: " <<
        tracker.process_model_->state_dimension() +
        tracker.process_model_->noise_dimension() +
        tracker.obsrv_model_->noise_dimension() << std::endl;

    Eigen::MatrixXd b;

    int max_cycles;
    int cycle = 0;
    ri::ReadParameter("max_cycles", max_cycles, nh);

    double t = 0.;
    double step;
    double a;
    ri::ReadParameter("step", step, nh);
    ri::ReadParameter("a", a, nh);

    std::cout << ">> ready to run ..." << std::endl;

    std::ofstream groundtruth;
    groundtruth.open ("/home/issac_local/groundtruth.txt", std::ios::out);
    std_msgs::Float32 y_msg;
    std_msgs::Float32 y_e_msg;
    ros::Publisher pub_y = nh.advertise<std_msgs::Float32>("/dev_test_example/y", 10000000);
    ros::Publisher pub_y_e = nh.advertise<std_msgs::Float32>("/dev_test_example/y_e", 10000000);

    fl::StandardGaussian<double> g;

    while(ros::ok())
    {
        //        groundtruth << t << " "
        //                    << y(0) << " "
        //                    << tracker.state_distr_.mean() << std::endl;

        t += step;
        y(0) = std::sin(a * t);

        tracker.filter(y);

        y_msg.data = y(0);
        y_e_msg.data = tracker.state_distr_.mean()(0, 0);

        pub_y.publish(y_msg);
        pub_y_e.publish(y_e_msg);

        if (max_cycles > 0 && cycle++ > max_cycles) break;

        ros::Duration(0.005).sleep();

        ros::spinOnce();
    }

    groundtruth.close();

    return 0;
}
