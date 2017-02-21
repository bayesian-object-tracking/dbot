/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
 * Probabilistic Object Tracking using a Range Camera
 * IEEE Intl Conf on Intelligent Robots and Systems, 2013
 * http://arxiv.org/abs/1505.00241
 *
 */

/**
 * \date May 2014
 * \file occlusion_model.h
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

// TODO: THIS IS JUST A LINEAR GAUSSIAN PROCESS WITH NO NOISE, SHOULD DISAPPEAR
namespace dbot
{

/**
 * \class OcclusionProcess
 *
 * \ingroup distributions
 * \ingroup transitions
 */
class OcclusionModel
{
public:
    // the prob of source being object given source was object one sec ago,
    // and prob of source being object given one sec ago source was not object
    OcclusionModel(double p_occluded_visible,
                          double p_occluded_occluded):
                                      p_occluded_visible_(p_occluded_visible),
                                      p_occluded_occluded_(p_occluded_occluded),
                                      c_(p_occluded_occluded_ - p_occluded_visible_),
                                      log_c_(std::log(c_)) { }

    virtual ~OcclusionModel() noexcept {}

    virtual void Condition(const double& delta_time,
                           const double& occlusion_probability,
                           const double& input = 0)
    {
        delta_time_ = delta_time;
        occlusion_probability_ = occlusion_probability;
    }

    virtual double MapStandardGaussian() const
    {
        double pow_c_time = std::exp(delta_time_*log_c_);

        double new_occlusion_probability =
                1. - (pow_c_time*(1.-occlusion_probability_) +
                    (1 - p_occluded_occluded_)*(pow_c_time-1.)/(c_-1.));

        if(new_occlusion_probability < 0.0 || new_occlusion_probability > 1.0)
        {
            if(std::fabs(c_ - 1.0) < 0.000000001)
            {
                new_occlusion_probability = occlusion_probability_;
            }
            else
            {
                std::cout << "unhandeled case in occlusion process mdoel " << std::endl;
                exit(-1);
            }
        }

        return new_occlusion_probability;
    }

private:
    // conditionals
    double occlusion_probability_, delta_time_;
    // parameters
    double p_occluded_visible_, p_occluded_occluded_, c_, log_c_;
};

}
