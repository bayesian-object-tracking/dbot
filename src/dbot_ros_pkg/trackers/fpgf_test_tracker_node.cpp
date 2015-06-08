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
#include <fl/model/process/joint_process_model.hpp>
#include <fl/model/process/linear_process_model.hpp>
#include <fl/filter/gaussian/unscented_transform.hpp>
#include <fl/filter/gaussian/monte_carlo_transform.hpp>
#include <fl/filter/gaussian/gaussian_filter_ukf.hpp>
#include <fl/filter/gaussian/gaussian_filter_factorized.hpp>

#include <dbot/utils/rigid_body_renderer.hpp>
#include <dbot/states/free_floating_rigid_bodies_state.hpp>
#include <dbot/models/process_models/brownian_object_motion_model.hpp>

#include <dbot_ros_pkg/utils/image_publisher.hpp>

#include <fl/util/profiling.hpp>

#include "dev_test_tracker/virtual_object.hpp"
#include "dev_test_tracker/vector_hashing.hpp"

#include "../fpf_test/squared_feature_policy.hpp"
#include "../fpf_test/depth_pixel_model.hpp"

#include <std_msgs/Float32.h>

using namespace fl;

class FilterContext
{
public:
    enum : signed int { PixelCount = Eigen::Dynamic };

    struct Arguments
    {
        double linear_accel_sigma;
        double angular_accel_sigma;
        double damping;
        double A_b;
        double v_sigma_b;
        double w_sigma;
        double bg_depth;
        double bg_sigma;
        double downsampling;
    };

public:
    /* ############################## */
    /* # Basic types                # */
    /* ############################## */
    typedef double Scalar;
    typedef FreeFloatingRigidBodiesState<> ObjectState;


    /* ############################## */
    /* # Observation Model          # */
    /* ############################## */
    typedef DepthPixelModel<ObjectState> DPixelModel;


    /* ############################## */
    /* # Process Model              # */
    /* ############################## */
    typedef BrownianObjectMotionModel<ObjectState> ProcessModel;


    /* ############################## */
    /* # Observation Param Model    # */
    /* ############################## */
    typedef typename Traits<DPixelModel>::Param Param;
    typedef LinearGaussianProcessModel<Param> ParamModel;


    /* ############################## */
    /* # Filter                     # */
    /* ############################## */
    typedef MonteCarloTransform<
                LinearPointCountPolicy<10>
            > PointTransform;

    typedef GaussianFilter<
                ProcessModel,
                Join<MultipleOf<Adaptive<DPixelModel, ParamModel>, PixelCount>>,
                PointTransform,
                SigmoidFeaturePolicy<>,
                Options<FactorizeParams>
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


    FilterContext(Arguments& args, VirtualObject<ObjectState>& object)
        : object_(object),
          filter_(
              create_filter(
                  create_process_model(args),
                  create_param_model(args),
                  create_pixel_model(args))),
          state_distr_(filter_.create_state_distribution()),
          zero_input_(filter_.process_model().input_dimension(), 1),
          y(filter_.obsrv_model().obsrv_dimension(), 1)
    {
        zero_input_.setZero();
    }


    void filter(std::vector<float>& y_vec)
    {
        const int y_vec_size = y_vec.size();

        y.resize(y_vec_size, 1);
        for (int i = 0; i < y_vec_size; ++i)
        {
            y(i, 0) = y_vec[i];
        }

        filter_.predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_.update(y, state_distr_, state_distr_);
    }

    FilterAlgo& algo() { return filter_; }

private:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    ProcessModel create_process_model(Arguments& args)
    {
        auto model = ProcessModel(1);

        Eigen::MatrixXd linear_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(args.linear_accel_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(args.angular_accel_sigma), 2);

        model.parameters(0,
                         object_.renderer->object_center(0).cast<double>(),
                         args.damping,
                         linear_acceleration_covariance,
                         angular_acceleration_covariance);

        return model;
    }

    ParamModel create_param_model(Arguments& args)
    {
        auto model = ParamModel();

        // set dynamics matrix and noise covariance
        model.A(model.A() * args.A_b);
        model.covariance(model.covariance() * std::pow(args.v_sigma_b, 2));

        return model;
    }

    DPixelModel create_pixel_model(Arguments& args)
    {
        auto model = DPixelModel(object_.renderer, ObjectState::BODY_SIZE);

        // set param initial value and noise covariance
        model.covariance(model.covariance() * std::pow(args.w_sigma, 2));
        model.bg_depth_ = args.bg_depth;
        model.bg_sigma_ = args.bg_sigma;

        return model;
    }

    FilterAlgo create_filter(ProcessModel process_model,
                             ParamModel param_model,
                             DPixelModel pixel_model)
    {
        int pixel_count = object_.res_rows * object_.res_cols;

        auto filter = FilterAlgo(process_model,
                                 param_model,
                                 pixel_model,
                                 PointTransform(),
                                 FeatureMapping(),
                                 pixel_count);

        return filter;
    }

public:
    VirtualObject<ObjectState>& object_;

    FilterAlgo filter_;
    StateDistribution state_distr_;
    Input zero_input_;
    Obsrv y;

    int pixel_count;
};


int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "dev_test_tracker");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Deduce filter types        # */
    /* ############################## */
    typedef FilterContext::Filter::State State;
    typedef FilterContext::Filter::Input Input;
    typedef FilterContext::Filter::Obsrv Obsrv;
    typedef FilterContext::Filter::StateDistribution StateDistribution;

    // read parameters
    std::string depth_image_topic;
    std::string point_cloud_topic;
    std::string camera_info_topic;

    std::cout << "reading parameters" << std::endl;
    ri::ReadParameter("camera_info_topic", camera_info_topic, nh);
    ri::ReadParameter("depth_image_topic", depth_image_topic, nh);

    point_cloud_topic = "/XTION/depth/points";
    camera_info_topic = "/XTION/depth/camera_info";

    FilterContext::Arguments args;
    ri::ReadParameter("downsampling", args.downsampling, nh);
    ri::ReadParameter("A_b", args.A_b, nh);
    ri::ReadParameter("v_sigma_b", args.v_sigma_b, nh);
    ri::ReadParameter("w_sigma", args.w_sigma, nh);
    ri::ReadParameter("linear_acceleration_sigma", args.linear_accel_sigma, nh);
    ri::ReadParameter("angular_acceleration_sigma", args.angular_accel_sigma, nh);
    ri::ReadParameter("damping", args.damping, nh);
    ri::ReadParameter("bg_depth", args.bg_depth, nh);
    ri::ReadParameter("bg_sigma", args.bg_sigma, nh);

    Eigen::Matrix3d camera_matrix =
            ri::GetCameraMatrix<double>(camera_info_topic, nh, 2.0);
    camera_matrix.topLeftCorner(2, 3) /= args.downsampling;

    // get observations from camera
    sensor_msgs::Image::ConstPtr ros_image =
            ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic,
                                                           nh,
                                                           ros::Duration(10.0));

    Obsrv obsrv = ri::Ros2Eigen<double>(*ros_image, args.downsampling);


    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    VirtualObject<FilterContext::ObjectState> object(nh);
    int pixel_count = object.res_rows * object.res_cols;

    FilterContext context(args, object);

    // init state distribution (FPF)
    auto& distr_a = std::get<0>(context.state_distr_.distributions());
    auto& distr_b = std::get<1>(context.state_distr_.distributions())
            .distributions();

    distr_a.mean(object.state);
    distr_a.covariance(distr_a.covariance() * 0.0001);
    for (int i = 0; i < pixel_count; ++i)
    {
        distr_b(i).covariance(distr_b(i).covariance() * 0.0001);
    }

    /* ############################## */
    /* # Images                     # */
    /* ############################## */
    fl::ImagePublisher ip(nh);
    std::vector<float> depth;


    Eigen::MatrixXd cov_bb = Eigen::MatrixXd::Zero(pixel_count, pixel_count);

    while(ros::ok())
    {
        /* ############################## */
        /* # Animate & Render           # */
        /* ############################## */
        object.animate();
        object.render(depth);

        /* ############################## */
        /* # Filter                     # */
        /* ############################## */

        INIT_PROFILING
        context.filter(depth);        
        MEASURE("filtering")

        /* ############################## */
        /* # Visualize                  # */
        /* ############################## */
        object.publish_marker(object.state, 1, 1, 0, 0);

        // publish pose (FPF)
        auto& distr_a = std::get<0>(context.state_distr_.distributions());
        object.publish_marker(distr_a.mean(), 2, 0, 1, 0);        
        PV(distr_a.covariance());

        for (int i = 0; i < pixel_count; ++i)
        {
            if (std::isinf(context.y(i)))
            {
                context.y(i) = 7;
            }
        }

        ip.publish(
            context.y,
            "/fpgf/observation",
            object.res_rows, object.res_cols);

        ip.publish(
            context.filter_.predicted_obsrv,
            "/fpgf/prediction",
            object.res_rows, object.res_cols);

        auto& distr_b = std::get<1>(context.state_distr_.distributions());
        auto exp_b = distr_b.mean();
        for (int i = 0; i < pixel_count; ++i)
        {
            exp_b(i) = context.algo()
                           .obsrv_model()
                           .local_obsrv_model()
                           .sigma(exp_b(i));
        }
        ip.publish(exp_b, "/fpgf/b", object.res_rows, object.res_cols);


        for (int i = 0; i < pixel_count; ++i)
        {
            cov_bb(i, i) = distr_b.distribution(i).covariance()(0);
        }
        ip.publish(cov_bb, "/fpgf/cov_bb", pixel_count, pixel_count);

        ros::spinOnce();
    }

    ros::spin();

    return 0;
}
