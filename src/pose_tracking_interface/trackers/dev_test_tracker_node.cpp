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
#include "dev_test_tracker/depth_observation_model.hpp"

class DevTestTracker
{
public:
    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef fl::BrownianObjectMotionModel<
                fl::FreeFloatingRigidBodiesState<>
            >  ProcessModel;

    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    // Holistic observation model
    typedef fl::DepthObservationModel<
                typename fl::Traits<ProcessModel>::State,
                typename fl::Traits<ProcessModel>::Scalar,
                Eigen::Dynamic,
                Eigen::Dynamic
            > ObsrvModel;

    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef fl::GaussianFilter<
                ProcessModel,
                ObsrvModel,
                fl::UnscentedTransform
            > FilterAlgo;

    typedef fl::FilterInterface<FilterAlgo> Filter;

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef typename Filter::State State;
    typedef typename Filter::Input Input;
    typedef typename Filter::Observation Obsrv;
    typedef typename Obsrv::Scalar Scalar;
    typedef typename Filter::StateDistribution StateDistribution;

    DevTestTracker(ros::NodeHandle& nh, VirtualObject<State>& object)
        : object(object),
          process_model_(create_process_model(nh)),
          obsrv_model_(create_obsrv_model(nh, process_model_)),
          filter_(create_filter(nh, process_model_, obsrv_model_)),
          state_distr_(filter_->process_model()->state_dimension()),
          zero_input_(Input::Zero(filter_->process_model()->input_dimension(), 1)),
          y(filter_->observation_model()->observation_dimension(), 1)
    {
        state_distr_.mean(object.state);
        state_distr_.covariance(state_distr_.covariance() * 0.0000000000001);

        ri::ReadParameter("inv_sigma", filter_->inv_sigma, nh);
        ri::ReadParameter("threshold", filter_->threshold, nh);
    }

    void filter(std::vector<float> y_vec)
    {
        const int y_vec_size = y_vec.size();

        Scalar y_i;
        for (int i = 0; i < y_vec_size; ++i)
        {
            y_i = (std::isinf(y_vec[i]) ? 7 : y_vec[i]);
            y(2*i, 0) = y_i * y_i;
            y(2*i+1, 0) = y_i;
        }

        filter_->predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_->update(y, state_distr_, state_distr_);

        obsrv_model_->clear_cache();
    }

public:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    /**
     * \return Process model instance
     */
    std::shared_ptr<ProcessModel> create_process_model(ros::NodeHandle& nh)
    {
        double linear_acceleration_sigma;
        double angular_acceleration_sigma;
        double damping;

        ri::ReadParameter("linear_acceleration_sigma",
                          linear_acceleration_sigma, nh);
        ri::ReadParameter("angular_acceleration_sigma",
                          angular_acceleration_sigma, nh);
        ri::ReadParameter("damping",
                          damping, nh);

        auto model = std::make_shared<ProcessModel>(1 /* one object */);

        Eigen::MatrixXd linear_acceleration_covariance =
                Eigen::MatrixXd::Identity(3, 3)
                * std::pow(double(linear_acceleration_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
                Eigen::MatrixXd::Identity(3, 3)
                * std::pow(double(angular_acceleration_sigma), 2);

        model->parameters(0,
                          object.renderer->object_center(0).cast<double>(),
                          damping,
                          linear_acceleration_covariance,
                          angular_acceleration_covariance);

        return model;
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

        return std::make_shared<ObsrvModel>(object.renderer,
                                            camera_sigma,
                                            model_sigma,
                                            process_model->state_dimension(),
                                            object.res_rows,
                                            object.res_cols);
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

        return std::make_shared<FilterAlgo>(
                    process_model,
                    obsrv_model,
                    std::make_shared<fl::UnscentedTransform>(ut_alpha, ut_beta, ut_kappa));
                    //std::make_shared<fl::RandomTransform>());
    }

public:
    VirtualObject<State>& object;
    std::shared_ptr<ProcessModel> process_model_;
    std::shared_ptr<ObsrvModel> obsrv_model_;
    Filter::Ptr filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;
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
    VirtualObject<DevTestTracker::State> object(nh);
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

    while(ros::ok())
    {
        std::cout <<  "==========================================" << std::endl;
        std::cout << tracker.state_distr_.mean().transpose() << std::endl;
        std::cout <<  "--------------------" << std::endl;
        std::cout << tracker.state_distr_.covariance() << std::endl;

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

        ip.publish(tracker.filter_->innovation, "/dev_test_tracker/innovation",
                   object.res_rows,
                   object.res_cols);

        ip.publish(tracker.filter_->prediction, "/dev_test_tracker/prediction",
                   object.res_rows,
                   object.res_cols);

//        ip.publish(tracker.filter_->invR, "/dev_test_tracker/inv_covariance",
//                   object.res_rows,
//                   object.res_cols);

        ros::spinOnce();

    }

    return 0;
}
