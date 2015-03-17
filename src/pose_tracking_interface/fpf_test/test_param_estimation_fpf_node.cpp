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

#include <fl/distribution/standard_gaussian.hpp>

#include "test_param_estimation_fpf.hpp"
#include <std_msgs/Float32.h>

int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "param_estimation_fpf");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */

    typedef GfContext<Options<NoOptions>, 1000> JGfContext;
    typedef GfContext<Options<FactorizeParams>, 100> FpfContext;

    FpfContext context_fpf(nh);
    FpfContext::Obsrv y(context_fpf.algo().obsrv_model().obsrv_dimension(), 1);

    JGfContext context_jgf(nh);

    /* ############################## */
    /* # Images                     # */
    /* ############################## */
    fl::ImagePublisher ip(nh);

    int max_pub = 10;
    int max_cycles;
    int cycle = 0;
    ri::ReadParameter("max_cycles", max_cycles, nh);

    std_msgs::Float32 x_msg;
    ros::Publisher pub_x_gt
        = nh.advertise<std_msgs::Float32>("/fpf/x_gt", 10000);
    ros::Publisher pub_fpf_x_e
        = nh.advertise<std_msgs::Float32>("/fpf/x_e", 10000);

    ros::Publisher pub_jgf_x_e
        = nh.advertise<std_msgs::Float32>("/jgf/x_e", 10000);

    double ground_truth;

    int error_pixels;
    int cycle_of_error;
    int error_duration;
    double error_value;
    bool gradual_error;
    int gradual_delay;
    int current_error_pixels = 0;
    bool factorize;
    bool print_states;
    bool const_error;
    ri::ReadParameter("cycle_of_error", cycle_of_error, nh);
    ri::ReadParameter("error_duration", error_duration, nh);
    ri::ReadParameter("error_value", error_value, nh);
    ri::ReadParameter("const_error", const_error, nh);
    ri::ReadParameter("error_pixels", error_pixels, nh);
    ri::ReadParameter("gradual_error", gradual_error, nh);
    ri::ReadParameter("gradual_delay", gradual_delay, nh);
    ri::ReadParameter("factorize", factorize, nh);
    ri::ReadParameter("print_states", print_states, nh);

    int pixel_count;
    double sim_w_sigma;
    ri::ReadParameter("pixel_count", pixel_count, nh);
    ri::ReadParameter("sim_w_sigma", sim_w_sigma, nh);

    std::vector<std::pair<std_msgs::Float32, ros::Publisher>> pubs_y;
    std::vector<std::pair<std_msgs::Float32, ros::Publisher>> pubs_fpf_b;
    std::vector<std::pair<std_msgs::Float32, ros::Publisher>> pubs_jgf_b;
    for (int i = 0; i < std::min(pixel_count, max_pub); ++i)
    {
        pubs_y.push_back(
            std::make_pair<std_msgs::Float32, ros::Publisher>(
                std_msgs::Float32(),
                nh.advertise<std_msgs::Float32>(
                    std::string("/fpf/y") + std::to_string(i),
                    10000)));

        pubs_fpf_b.push_back(
            std::make_pair<std_msgs::Float32, ros::Publisher>(
                std_msgs::Float32(),
                nh.advertise<std_msgs::Float32>(
                    std::string("/fpf/b") + std::to_string(i),
                    10000)));

        pubs_jgf_b.push_back(
            std::make_pair<std_msgs::Float32, ros::Publisher>(
                std_msgs::Float32(),
                nh.advertise<std_msgs::Float32>(
                    std::string("/jgf/b") + std::to_string(i),
                    10000)));
    }

    double t = 0.;
    //double step =  8. * 2. * M_PI / double(max_cycles);
    double step =  8 * 2. * M_PI / double(max_cycles);


    fl::Gaussian<FpfContext::Obsrv> g(
            context_fpf.algo().obsrv_model().obsrv_dimension());
    g.covariance(g.covariance() * std::pow(sim_w_sigma, 2));

    fl::StandardGaussian<double> g_failing;

    std::cout << ">> ready to run ..." << std::endl;

    double v_sigma_a;
    double v_sigma_b;
    ri::ReadParameter("v_sigma_a", v_sigma_a, nh);
    ri::ReadParameter("v_sigma_b", v_sigma_b, nh);

    // set initial state distribution of fpf
    auto& fpf_distr_a = std::get<0>(context_fpf.state_distr_.distributions());
    auto& fpf_distr_b = std::get<1>(context_fpf.state_distr_.distributions()).distributions();
    fpf_distr_a.covariance(fpf_distr_a.covariance() * v_sigma_a);
    for (int i = 0; i < pixel_count; ++i)
    {
        fpf_distr_b(i).covariance(fpf_distr_b(i).covariance() * v_sigma_b);
    }

    // set initial state distribution of jgf
    int dim_a = fpf_distr_a.dimension();
    int dim_b = std::get<1>(context_fpf.state_distr_.distributions()).dimension();
    auto jgf_cov = context_jgf.state_distr_.covariance();

    jgf_cov.block(0, 0, dim_a, dim_a) =
        jgf_cov.block(0, 0, dim_a, dim_a) * v_sigma_a;

    jgf_cov.block(dim_a, dim_a, dim_b, dim_b) =
        jgf_cov.block(dim_a, dim_a, dim_b, dim_b) * v_sigma_b;

    context_jgf.state_distr_.covariance(jgf_cov);

    while(ros::ok())
    {
        /* ############################ */
        /* # create measurements      # */
        /* ############################ */

        ground_truth = 0.5 * std::sin(t) +
                       0.3 * std::sin(0.5 * t) +
                       0.2 * std::sin(1.5 * t) -
                       0.2 * std::cos(2 * t) +
                       0.3 * std::sin(1. + 2 * t);

//        ground_truth = std::sin(t);

        t += step;
        y.setOnes();
        y = y * ground_truth + g.sample();

        /* ############################ */
        /* # introduce errors         # */
        /* ############################ */

        if (cycle_of_error < cycle && cycle_of_error + error_duration > cycle)
        {
            /* compute the number of error pixels */
            if (!gradual_error)
            {
                current_error_pixels = error_pixels;
            }
            else
            {
                current_error_pixels =
                    std::min((cycle - cycle_of_error) / gradual_delay + 1,
                             error_pixels);

                std::cout << "current_error_pixels = "
                          << current_error_pixels << std::endl;
            }

            /* set errors */
            for (int i = 0; i < current_error_pixels && i < pixel_count; ++i)
            {
                if (const_error)
                {
                    y(i) = error_value;
                }
                else
                {
                    y(i) = g_failing.sample() * error_value;
                }
            }
        }

        /* ############################ */
        /* # Filter                   # */
        /* ############################ */
        context_fpf.filter(y);
        context_jgf.filter(y);

        /* ############################ */
        /* # Approx JGF Covariance    # */
        /* ############################ */
        if (factorize)
        {
            auto iden = context_jgf.state_distr_.covariance();
            iden.setIdentity();
            context_jgf.state_distr_.covariance(
                context_jgf.state_distr_.covariance().cwiseProduct(iden));
        }

        if (print_states)
        {
            PV(context_fpf.state_distr_.mean().transpose());
            PV(context_jgf.state_distr_.mean().transpose());
            PV(context_fpf.state_distr_.covariance());
            PV(context_jgf.state_distr_.covariance());
        }

        /* ############################ */
        /* # Visualize                # */
        /* ############################ */
        // y
        for (int i = 0; i < std::min(pixel_count, max_pub); ++i)
        {
            pubs_y[i].first.data = y(2 * i);
            pubs_y[i].second.publish(pubs_y[i].first);
        }

        auto&& fpf_distr_a = std::get<0>(
                    context_fpf.state_distr_.distributions());

        auto&& jgf_state = context_jgf.state_distr_.mean();

        // a
        x_msg.data = fpf_distr_a.mean()(0);
        pub_fpf_x_e.publish(x_msg);

        x_msg.data = jgf_state(0);
        pub_jgf_x_e.publish(x_msg);

        // b
        auto&& distr_b = std::get<1>(context_fpf.state_distr_.distributions());
        for (int i = 0; i < std::min(pixel_count, max_pub); ++i)
        {
            // publish fpf b
            double b = distr_b.distribution(i).mean()(0);
            pubs_fpf_b[i].first.data = context_fpf.algo()
                                       .obsrv_model()
                                       .local_obsrv_model()
                                       .sigma(b);
            pubs_fpf_b[i].second.publish(pubs_fpf_b[i].first);

            // publish jgf b
            pubs_jgf_b[i].first.data = context_jgf.algo()
                                           .obsrv_model()
                                           .local_obsrv_model()
                                           .sigma(jgf_state(1 + i));
            pubs_jgf_b[i].second.publish(pubs_jgf_b[i].first);
        }

//        // y
//        x_msg.data = y.mean();
//        pub_x_avrg.publish(x_msg);

        // ground truth
        x_msg.data = ground_truth;
        pub_x_gt.publish(x_msg);

        ip.publish(y, "/fpf/obsrv", 1, y.rows());

        if (max_cycles > 0 && cycle++ > max_cycles) break;

        ros::Duration(0.01).sleep();
        ros::spinOnce();
    }

    std::cout << ">> terminating ..." << std::endl;

    return 0;
}
;
