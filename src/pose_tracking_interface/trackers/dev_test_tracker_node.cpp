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

#include <std_msgs/Header.h>
#include <ros/ros.h>
#include <ros/package.h>

#include <fl/util/meta.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/random_transform.hpp>

#include <pose_tracking/utils/rigid_body_renderer.hpp>
#include <pose_tracking/states/free_floating_rigid_bodies_state.hpp>
#include <pose_tracking/models/process_models/brownian_object_motion_model.hpp>

#include <pose_tracking_interface/utils/image_publisher.hpp>

#include <fl/util/profiling.hpp>

#include "dev_test_tracker/virtual_object.hpp"
#include "dev_test_tracker/linear_depth_observation_model.hpp"
#include "dev_test_tracker/pixel_observation_model.hpp"
#include "dev_test_tracker/depth_observation_model.hpp"
#include "dev_test_tracker/dual_process_model.hpp"

class DevTestTracker
{
public:
    /* ############################## */
    /* # Basic types                # */
    /* ############################## */
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 1, 1> ParameterState;
    typedef fl::FreeFloatingRigidBodiesState<> ObjectState;

    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef fl::BrownianObjectMotionModel<ObjectState> PoseProcessModel;

    typedef fl::LinearGaussianProcessModel<ParameterState> ParameterProcessModel;
    typedef fl::FactorizedIIDProcessModel<
                ParameterProcessModel
            > ParametersProcessModel;

    typedef fl::DualProcessModel<
                PoseProcessModel,
                ParametersProcessModel
            > ProcessModel;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    /* Pixel Model */
    typedef fl::PixelObservationModel<
                Scalar
            > PixelObsrvModel;

    /* Camera Model(Pixel Model) */
    typedef fl::FactorizedIIDObservationModel<
                PixelObsrvModel
            > CameraObservationModel;

    /* Depth Model(Camera Model(Pixel Model)) */
    typedef fl::DepthObservationModel<
                CameraObservationModel,
                typename fl::Traits<ProcessModel>::State
            > ObsrvModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObsrvModel,
                //fl::UnscentedTransform
                fl::RandomTransform
            > FilterAlgo;

    typedef fl::FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::State State;
    typedef typename Filter::Input Input;
    typedef typename Filter::Observation Obsrv;
    typedef typename Filter::StateDistribution StateDistribution;

    DevTestTracker(ros::NodeHandle& nh, VirtualObject<ObjectState>& object)
        : object(object),

          process_model_(create_process_model(nh)),
          obsrv_model_(create_obsrv_model(nh, process_model_)),
          filter_(create_filter(nh, process_model_, obsrv_model_)),

          state_distr_(filter_->process_model()->state_dimension()),
          zero_input_(Input::Zero(filter_->process_model()->input_dimension(), 1)),
          y(filter_->observation_model()->observation_dimension(), 1)
    {
        State initial_state = State::Zero(filter_->process_model()->state_dimension(), 1);
        initial_state.topRows(object.state.rows()) = object.state;

        state_distr_.mean(initial_state);
        state_distr_.covariance(state_distr_.covariance() * 1.e-12);

        ri::ReadParameter("inv_sigma", filter_->inv_sigma, nh);
        ri::ReadParameter("threshold", filter_->threshold, nh);
        ri::ReadParameter("print_ukf_details", filter_->print_details, nh);

        ri::ReadParameter("error_pixel", error_pixel, nh);
        ri::ReadParameter("error_pixel_depth", error_pixel_depth, nh);
    }

    void filter(std::vector<float>& y_vec)
    {
        const int y_vec_size = y_vec.size();


        if (error_pixel >= 0 && error_pixel < y_vec_size)
        {
            y_vec[error_pixel] = error_pixel_depth;
        }
        else if (error_pixel >= y_vec_size)
        {
            std::cout << "error pixel index overflow at " << error_pixel
                      << ", max: " << y_vec_size << std::endl;
        }

        Scalar y_i;
        for (int i = 0; i < y_vec_size; ++i)
        {
            y_i = (std::isinf(y_vec[i]) ? 7 : y_vec[i]);
            y(2 * i) = y_i;
            y(2 * i + 1) = y_i * y_i;
        }                

        filter_->predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_->update(y, state_distr_, state_distr_);

        obsrv_model_->clear_cache();
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
        double linear_accel_sigma;
        double angular_accel_sigma;
        double damping;
        double b_sigma;
        double sigma_decay;

        ri::ReadParameter("linear_acceleration_sigma", linear_accel_sigma, nh);
        ri::ReadParameter("angular_acceleration_sigma", angular_accel_sigma, nh);
        ri::ReadParameter("damping", damping, nh);
        ri::ReadParameter("b_sigma", b_sigma, nh);
        ri::ReadParameter("sigma_decay", sigma_decay, nh);

        auto pose_model = std::make_shared<PoseProcessModel>(1 /* 1 object */);

        Eigen::MatrixXd linear_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(linear_accel_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(angular_accel_sigma), 2);

        pose_model->parameters(
            0,
            object.renderer->object_center(0).cast<double>(),
            damping,
            linear_acceleration_covariance,
            angular_acceleration_covariance);

        typedef typename fl::Traits<ParameterProcessModel>::SecondMoment ParamCov;

        auto parameter_model =
            std::make_shared<ParameterProcessModel>(
                ParamCov::Identity() * (b_sigma * b_sigma));

        parameter_model->A(parameter_model->A() * sigma_decay);

        auto parameters_model = std::make_shared<ParametersProcessModel>(
                    parameter_model,
                    object.res_rows * object.res_cols);

        return std::make_shared<ProcessModel>(pose_model, parameters_model);
    }

    /**
     * \return Observation model instance
     */
    std::shared_ptr<ObsrvModel> create_obsrv_model(
            ros::NodeHandle& nh,
            std::shared_ptr<ProcessModel> process_model)
    {
        double model_sigma;
        double camera_sigma;
        ri::ReadParameter("model_sigma", model_sigma, nh);
        ri::ReadParameter("camera_sigma", camera_sigma, nh);

        double pixel_variance =
            (camera_sigma * camera_sigma) + (model_sigma * model_sigma);

        return std::make_shared<ObsrvModel>
        (
            std::make_shared<CameraObservationModel>
            (
                std::make_shared<PixelObsrvModel>(pixel_variance),
                (object.res_rows * object.res_cols)
            ),
            object.renderer,
            process_model->pose_process_model()->state_dimension(),
            process_model->state_dimension()
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
        double ut_alpha;
        double ut_beta;
        double ut_kappa;
        ri::ReadParameter("ut_alpha", ut_alpha, nh);
        ri::ReadParameter("ut_beta", ut_beta, nh);
        ri::ReadParameter("ut_kappa", ut_kappa, nh);

        return std::make_shared<FilterAlgo>
        (
            process_model,
            obsrv_model,
            //std::make_shared<fl::UnscentedTransform>(ut_alpha, ut_beta, ut_kappa));
            std::make_shared<fl::RandomTransform>()
        );
    }

public:
    VirtualObject<ObjectState>& object;
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
    Filter::Ptr filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;


    int error_pixel;
    double error_pixel_depth;
};


int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "dev_test_tracker");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    VirtualObject<DevTestTracker::ObjectState> object(nh);
    DevTestTracker tracker(nh, object);

    /* ############################## */
    /* # Images                     # */
    /* ############################## */
    fl::ImagePublisher ip(nh);
    std::vector<float> depth;
    Eigen::MatrixXd image_vector(object.res_rows * object.res_cols, 1);

    std::cout << ">> initial state " << std::endl;
    std::cout << tracker.state_distr_.mean().transpose() << std::endl;

    std::cout << ">> setup: " << std::endl;    
    std::cout << ">> state dimension: " << tracker.process_model_->state_dimension() << std::endl;
    std::cout << ">> noise dimension: " << tracker.process_model_->noise_dimension() << std::endl;
    std::cout << ">> obsrv dimension: " << tracker.obsrv_model_->observation_dimension() << std::endl;
    std::cout << ">> obsrv noise dimension: " << tracker.obsrv_model_->noise_dimension() << std::endl;
    std::cout << ">> obsrv state dimension: " << tracker.obsrv_model_->state_dimension() << std::endl;

    std::cout << ">> sigma point number: " <<
        tracker.process_model_->state_dimension() +
        tracker.process_model_->noise_dimension() +
        tracker.obsrv_model_->noise_dimension() << std::endl;

    Eigen::MatrixXd b;



    int max_cycles;
    int cycle = 0;
    ri::ReadParameter("max_cycles", max_cycles, nh);

    while(ros::ok())
    {
//        std::cout <<  "==========================================" << std::endl;
//        std::cout << tracker.state_distr_.mean().transpose() << std::endl;
//        std::cout <<  "--------------------" << std::endl;
//       std::cout << tracker.state_distr_.covariance() << std::endl;


        b = tracker.state_distr_.mean().bottomRows(
                tracker.process_model_->parameters_process_model()->state_dimension());

        /* ############################## */
        /* # Animate & Render           # */
        /* ############################## */
        object.animate();
        object.render(depth);

        /* ############################## */
        /* # Filter                     # */
        /* ############################## */
        INIT_PROFILING
        tracker.filter(depth);
        MEASURE("filter")

        /* ############################## */
        /* # Visualize                  # */
        /* ############################## */
        object.publish_marker(object.state, 1, 1, 0, 0);
        object.publish_marker(tracker.state_distr_.mean(), 2, 0, 1, 0);

        image_vector.setZero();
        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = (std::isinf(depth[i]) ? 0 : depth[i]);
        }
        ip.publish(image_vector, "/dev_test_tracker/observation",
                   object.res_rows,
                   object.res_cols);

        /* ############################## */

        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = tracker.filter_->innovation(2 * i, 0);
        }
        ip.publish(image_vector, "/dev_test_tracker/innovation",
                   object.res_rows,
                   object.res_cols);

        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = tracker.filter_->innovation(2 * i + 1, 0);
        }
        ip.publish(image_vector, "/dev_test_tracker/innovation_pow2",
                   object.res_rows,
                   object.res_cols);

        /* ############################## */

        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = tracker.filter_->prediction(2*i, 0);
        }
        ip.publish(image_vector, "/dev_test_tracker/prediction",
                   object.res_rows,
                   object.res_cols);


        for (size_t i = 0; i < depth.size(); ++i)
        {
            image_vector(i, 0) = tracker.filter_->prediction(2 * i + 1, 0);
        }
        ip.publish(image_vector, "/dev_test_tracker/prediction_pow2",
                   object.res_rows,
                   object.res_cols);

        /* ############################## */

        ip.publish(b, "/dev_test_tracker/b",
                   object.res_rows,
                   object.res_cols);

        ros::spinOnce();


        if (max_cycles > 0 && cycle > max_cycles)
        {
            break;
        }
        cycle++;
    }

    return 0;
}
