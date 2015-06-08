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

#include<iostream>
#include <fstream>

#include "dev_test_example.hpp"

int main (int argc, char **argv)
{
    /* ############################## */
    /* # Setup ros                  # */
    /* ############################## */
    ros::init(argc, argv, "dev_test_plots");
    ros::NodeHandle nh("~");

    /* ############################## */
    /* # Setup object and tracker   # */
    /* ############################## */
    DevTestExample tracker(nh);

    const int dim_y = tracker.obsrv_model_->obsrv_dimension();
    const int pixels = tracker.pixels;

    DevTestExample::Observation y(dim_y, 1);

    int max_cycles;

    int error_pixels;
    int cycle_of_error;
    int error_duration;
    double error_value;
    double w_sigma;
    double sim_w_sigma;
    double max_w_sigma;

    ri::ReadParameter("max_cycles", max_cycles, nh);
    ri::ReadParameter("cycle_of_error", cycle_of_error, nh);
    ri::ReadParameter("error_duration", error_duration, nh);
    ri::ReadParameter("error_value", error_value, nh);
    ri::ReadParameter("error_pixels", error_pixels, nh);
    ri::ReadParameter("w_sigma", w_sigma, nh);
    ri::ReadParameter("sim_w_sigma", sim_w_sigma, nh);
    ri::ReadParameter("max_w_sigma", max_w_sigma, nh);

    double t = 0.;
    double step = 2. * M_PI / double(max_cycles);

    fl::Gaussian<DevTestExample::Observation> g(dim_y);
    g.covariance(g.covariance() * sim_w_sigma * sim_w_sigma);




    // plot config
    double range_step;
    double plot_range_x_min;
    double plot_range_x_max;
    double k;
    ri::ReadParameter("range_step", range_step, nh);
    ri::ReadParameter("plot_range_x_min", plot_range_x_min, nh);
    ri::ReadParameter("plot_range_x_max", plot_range_x_max, nh);
    ri::ReadParameter("k", tracker.obsrv_model_->pixel_observation_model()->k_, nh);

    t = plot_range_x_min;
    std::ofstream sigma_plots("/home/issac_local/sigma_curves.txt");

    if (!sigma_plots) std::cout << "creating file failed;" << std::endl;

    sigma_plots << "x s0 s1 s2 s3 s4\n";

    while(t < plot_range_x_max)
    {
        sigma_plots << t << " "
                    << tracker.obsrv_model_->pixel_observation_model()->sigma(t, 0) << " "
                    << tracker.obsrv_model_->pixel_observation_model()->sigma(t, 1) << " "
                    << tracker.obsrv_model_->pixel_observation_model()->sigma(t, 2) << " "
                    << tracker.obsrv_model_->pixel_observation_model()->sigma(t, 3) << " "
                    << tracker.obsrv_model_->pixel_observation_model()->sigma(t, 4) << "\n";
        t += range_step;
    }
    sigma_plots.close();

//    int cycle = 0;
//    double ground_truth;
//    while(ros::ok())
//    {
//        ground_truth = std::sin(t);

//        t += step;
//        y.setOnes();
//        y = y * ground_truth + g.sample();
//        for (int i = 0; i < pixels; ++i) y(2 * i + 1) = y(2 * i) * y(2 * i);


//        if (cycle_of_error < cycle && cycle_of_error + error_duration > cycle)
//        {
//            for (int i = 0; i < error_pixels && i < pixels; ++i)
//            {
//                y(2 * i) = error_value;
//                y(2 * i + 1) = y(2 * i) * y(2 * i);
//            }
//        }

//        tracker.filter(y);

//        if (max_cycles > 0 && cycle++ > max_cycles) break;
//    }

    return 0;
}
