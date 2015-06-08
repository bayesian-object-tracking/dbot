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
                LinearPointCountPolicy<8>
            > PointTransform;


//   typedef UnscentedTransform PointTransform;

//    typedef IdentityFeaturePolicy<> FeaturePolicy;
    //typedef SquaredFeaturePolicy<> FeaturePolicy;
    typedef SigmoidFeaturePolicy<> FeaturePolicy;

    typedef GaussianFilter<
                ProcessModel,
                Join<MultipleOf<Adaptive<DPixelModel, ParamModel>, PixelCount>>,
                PointTransform,
                FeaturePolicy,
                Options<FactorizeParams>
//                Options<NoOptions>
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

    FilterContext(ros::NodeHandle& nh, VirtualObject<ObjectState>& object)
        : object_(object),
          filter_(
              create_filter(nh,
                  create_process_model(nh),
                  create_param_model(nh),
                  create_pixel_model(nh))),
          state_distr_(filter_.create_state_distribution()),
          zero_input_(filter_.process_model().input_dimension(), 1),
          y(filter_.obsrv_model().obsrv_dimension(), 1)
    {
        zero_input_.setZero();

        ri::ReadParameter("bg_depth_sim", bg_depth, nh);
        ri::ReadParameter("w_sigma_sim", w_sigma_sim, nh);
        ri::ReadParameter("error_delay", error_delay, nh);
        ri::ReadParameter("error_pixels", error_pixels, nh);
        ri::ReadParameter("error_pixel_depth", error_pixel_depth, nh);
    }


    void filter(std::vector<float>& y_vec)
    {
        const int y_vec_size = y_vec.size();

        assert(y.rows() == y_vec_size);

        int error_pixels = this->error_pixels;

        for (int i = 0; i < y_vec_size; ++i)
        {
            y(i, 0) = (std::isinf(y_vec[i]) ? bg_depth : y_vec[i]);
            y(i, 0) += g_noise.sample() * w_sigma_sim;

            if (!error_delay &&
                !std::isinf(y_vec[i]) &&
                error_pixels-- > 0)
            {
                std::cout << "errors in da house" << std::endl;
                y(i, 0) += error_pixel_depth;
            }
        }

        if (error_delay > 0) error_delay--;

        filter_.predict(0.033, zero_input_, state_distr_, state_distr_);
        filter_.update(y, state_distr_, state_distr_);
    }

    FilterAlgo& algo() { return filter_; }

private:
    /* ############################## */
    /* # Factory Functions          # */
    /* ############################## */

    ProcessModel create_process_model(ros::NodeHandle& nh)
    {
        double linear_accel_sigma;
        double angular_accel_sigma;
        double damping;

        ri::ReadParameter("linear_acceleration_sigma", linear_accel_sigma, nh);
        ri::ReadParameter("angular_acceleration_sigma", angular_accel_sigma, nh);
        ri::ReadParameter("damping", damping, nh);

        auto model = ProcessModel(1);

        Eigen::MatrixXd linear_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(linear_accel_sigma), 2);

        Eigen::MatrixXd angular_acceleration_covariance =
            Eigen::MatrixXd::Identity(3, 3)
            * std::pow(double(angular_accel_sigma), 2);

        model.parameters(0,
                         object_.renderer->object_center(0).cast<double>(),
                         damping,
                         linear_acceleration_covariance,
                         angular_acceleration_covariance);

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

    DPixelModel create_pixel_model(ros::NodeHandle& nh)
    {
        double w_sigma;
        ri::ReadParameter("w_sigma", w_sigma, nh);

        auto model = DPixelModel(object_.renderer, ObjectState::BODY_SIZE);

        // set param initial value and noise covariance
        model.covariance(model.covariance() * std::pow(w_sigma, 2));

        ri::ReadParameter("bg_depth", model.bg_depth_, nh);
        ri::ReadParameter("bg_sigma", model.bg_sigma_, nh);

        return model;
    }

    FilterAlgo create_filter(ros::NodeHandle& nh,
                             ProcessModel process_model,
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
    StandardGaussian<double> g_noise;
    double w_sigma_sim;

    int pixel_count;
    int error_delay;
    int error_pixels;
    double error_pixel_depth;
    double bg_depth;
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

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    VirtualObject<FilterContext::ObjectState> object(nh);
    int pixel_count = object.res_rows * object.res_cols;

    FilterContext context(nh, object);

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

//    Eigen::MatrixXd b;

    int max_cycles;
    int cycle = 0;
    bool step_through;
    ri::ReadParameter("max_cycles", max_cycles, nh);
    ri::ReadParameter("step_through", step_through, nh);

    std::vector<std::pair<std_msgs::Float32, ros::Publisher>> pubs_fpf_a;
    std::vector<std::pair<std_msgs::Float32, ros::Publisher>> pubs_fpf_b;
    for (int i = 0; i < std::min(pixel_count, 50); ++i)
    {
        pubs_fpf_b.push_back(
            std::make_pair<std_msgs::Float32, ros::Publisher>(
                std_msgs::Float32(),
                nh.advertise<std_msgs::Float32>(
                    std::string("/fpf/b") + std::to_string(i),
                    10000)));
    }

    for (int i = 0; i < distr_a.dimension(); ++i)
    {
        pubs_fpf_a.push_back(
            std::make_pair<std_msgs::Float32, ros::Publisher>(
                std_msgs::Float32(),
                nh.advertise<std_msgs::Float32>(
                    std::string("/fpf/a") + std::to_string(i),
                    10000)));
    }

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
        ip.publish(context.y,
                   "/dev_test_tracker/observation",
                   object.res_rows, object.res_cols);

        ip.publish(context.filter_.predicted_obsrv,
                   "/dev_test_tracker/prediction",
                   object.res_rows, object.res_cols);

        auto& distr_b = std::get<1>(context.state_distr_.distributions());

        auto exp_b = distr_b.mean();
        for (int i = 0; i < std::min(pixel_count, 50); ++i)
        {
            exp_b(i) =
                context.algo().obsrv_model().local_obsrv_model().sigma(exp_b(i));
            pubs_fpf_b[i].first.data = exp_b(i);
            pubs_fpf_b[i].second.publish(pubs_fpf_b[i].first);
        }

        for (int i = 0; i < distr_a.dimension(); ++i)
        {
            pubs_fpf_a[i].first.data = distr_a.mean()(i);
            pubs_fpf_a[i].second.publish(pubs_fpf_a[i].first);
        }

        exp_b = distr_b.mean();
        for (int i = 0; i < pixel_count; ++i)
        {
            exp_b(i) = context.algo().obsrv_model().local_obsrv_model().sigma(exp_b(i));
        }
        ip.publish(exp_b,
                   "/dev_test_tracker/b",
                   object.res_rows, object.res_cols);


//        std::cout  << "cov_bb \n";
        for (int i = 0; i < pixel_count; ++i)
        {
            cov_bb(i, i) = distr_b.distribution(i).covariance()(0);
//            std::cout << cov_bb(i, i) << "  ";
        }
//        std::cout  << "\n";
        ip.publish(cov_bb,
                   "/dev_test_tracker/cov_bb",
                   pixel_count, pixel_count);

        ip.publish(distr_a.covariance(),
                   "/dev_test_tracker/cov_aa",
                   12, 12);

        ros::spinOnce();

        if (step_through)
        {
            ros::Duration(0.2).sleep();
            ros::spinOnce();

            std::cin.get();
        }

        if (max_cycles > 0 && cycle > max_cycles)
        {
            break;
        }
        cycle++;
    }

    return 0;
}
