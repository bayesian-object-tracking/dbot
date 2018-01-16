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
 * \file kinect_pixel_model.h
 * \author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <dbot/traits.h>
#include <iostream>

namespace dbot
{
// Forward declarations
class KinectPixelModel;

namespace internal
{
/**
 * KinectSensor distribution traits specialization
 * \internal
 */
template <>
struct Traits<KinectPixelModel>
{
    typedef double Scalar;
    typedef double Observation;
};
}

/**
 * \class KinectSensor
 *
 * \ingroup distributions
 * \ingroup sensors
 */
class KinectPixelModel
{
public:
    typedef typename internal::Traits<KinectPixelModel>::Scalar Scalar;
    typedef
        typename internal::Traits<KinectPixelModel>::Observation Observation;

    KinectPixelModel(Scalar tail_weight = 0.01,
                     Scalar model_sigma = 0.003,
                     Scalar sigma_factor = 0.00142478,
                     Scalar half_life_depth = 1.0,
                     Scalar max_depth = 6.0)
        : lambda_(-log(0.5) / half_life_depth),
          tail_weight_(tail_weight),
          model_sigma_(model_sigma),
          sigma_factor_(sigma_factor),
          max_depth_(max_depth)
    {
    }

    virtual ~KinectPixelModel() noexcept {}
    virtual Scalar Probability(const Observation& observation) const
    {
        // todo: if the prediction is infinite, the prob should not depend on
        // visibility. it does not matter
        // for the algorithm right now, but it should be changed
        Scalar probability;
        Scalar sigma = model_sigma_ + sigma_factor_ * observation * observation;
        if (!occlusion_)
        {
            if (std::isinf(prediction_))  // if the prediction_ is infinite we return
                                     // the limit
                probability = tail_weight_ / max_depth_;
            else
                probability = tail_weight_ / max_depth_ +
                              (1 - tail_weight_) *
                                  std::exp(-(pow(prediction_ - observation, 2) /
                                             (2 * sigma * sigma))) /
                                  (sqrt(2 * M_PI) * sigma);
        }
        else
        {
            if (std::isinf(prediction_))  // if the prediction_ is infinite we return
                                     // the limit
            {
                probability =
                    tail_weight_ / max_depth_ +
                    (1 - tail_weight_) * lambda_ *
                        std::exp(0.5 * lambda_ *
                                 (-2 * observation + lambda_ * sigma * sigma));
            }
            else
            {
                probability = tail_weight_ / max_depth_ +
                              (1 - tail_weight_) * lambda_ *
                                  std::exp(0.5 * lambda_ *
                                           (2 * prediction_ - 2 * observation +
                                            lambda_ * sigma * sigma)) *
                                  (1 + erf((prediction_ - observation +
                                            lambda_ * sigma * sigma) /
                                           (sqrt(2) * sigma))) /
                                  (2 * (std::exp(prediction_ * lambda_) - 1));
            }
        }

        return probability;
    }

    virtual Scalar LogProbability(const Observation& observation) const
    {
        return std::log(Probability(observation));
    }

    virtual void Condition(const Scalar& prediction, const bool& occlusion)
    {
        prediction_ = prediction;
        occlusion_ = occlusion;
    }

private:
    const Scalar lambda_, tail_weight_, model_sigma_, sigma_factor_, max_depth_;

    Scalar prediction_;
    bool occlusion_;
};
}
