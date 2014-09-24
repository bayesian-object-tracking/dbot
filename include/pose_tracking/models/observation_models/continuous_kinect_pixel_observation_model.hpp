/*************************************************************************
This software allows for filtering in high-dimensional observation and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/

#ifndef POSE_TRACKING_MODELS_OBSERVATION_MODELS_CONTINUOUS_KINECT_PIXEL_OBSERVATION_MODEL_HPP
#define POSE_TRACKING_MODELS_OBSERVATION_MODELS_CONTINUOUS_KINECT_PIXEL_OBSERVATION_MODEL_HPP

#include <cmath>
#include <Eigen/Dense>

#include <fast_filtering/distributions/interfaces/evaluation.hpp>
#include <fast_filtering/distributions/exponential_distribution.hpp>
#include <fast_filtering/distributions/uniform_distribution.hpp>
#include <fast_filtering/distributions/truncated_gaussian.hpp>

#include <pose_tracking/models/observation_models/kinect_pixel_observation_model.hpp>
#include <fast_filtering/utils/helper_functions.hpp>

namespace ff
{

/**
 * \class KinectObservationModel
 *
 * \ingroup distributions
 * \ingroup observation_models
 */
class ContinuousKinectPixelObservationModel:
        public GaussianMap<double, Eigen::Matrix<double, 5, 1> >,
        public Evaluation<double, double>
{
public:
    typedef Eigen::Matrix<double, 5, 1> Noise;
    typedef double Scalar;
    typedef double Observation;

    ContinuousKinectPixelObservationModel(Scalar sensor_failure_probability = 0.01,
                                          Scalar object_model_sigma = 0.003,
                                          Scalar sigma_factor = 0.00142478,
                                          Scalar half_life_depth = 1.0,
                                          Scalar max_depth = 6.0,
                                          Scalar min_depth = 0.0)
        :occluded_observation_model_(sensor_failure_probability,
                                     object_model_sigma,
                                     sigma_factor,
                                     half_life_depth,
                                     max_depth),
          visible_observation_model_(occluded_observation_model_),
          exponential_distribution_(-log(0.5)/half_life_depth, min_depth),
          object_model_noise_(0.0, object_model_sigma),
          sensor_failure_distribution_(min_depth, max_depth),
          lambda_(-log(0.5)/half_life_depth),
          sensor_failure_probability_(sensor_failure_probability),
          object_model_sigma_(object_model_sigma),
          sigma_factor_(sigma_factor),
          max_depth_(max_depth),
          min_depth_(min_depth){ }


    virtual ~ContinuousKinectPixelObservationModel() {}

    virtual Observation MapStandardGaussian(const Noise& sample) const
    {
        // object model inaccuracies
        Scalar object_depth = rendered_depth_
                + object_model_noise_.MapStandardGaussian(sample(0));

        // if the depth is outside of the range we return nan
        if(object_depth < min_depth_ || object_depth > max_depth_)
            return std::numeric_limits<Observation>::quiet_NaN();

        // occlusion
        bool occluded = uniform_distribution_.MapStandardGaussian(sample(1))
                <= occlusion_probability_ ? true : false;
        Scalar true_depth;
        if(occluded)
        {
            true_depth = exponential_distribution_.
                                MapStandardGaussian(sample(2), object_depth);
        }
        else
        {
            true_depth = object_depth;
        }

        // measurement noise
        bool sensor_failure =
                uniform_distribution_.MapStandardGaussian(sample(3))
                <= sensor_failure_probability_ ? true : false;
        Scalar measured_depth;
        if(sensor_failure)
        {
            measured_depth =
                    sensor_failure_distribution_.MapStandardGaussian(sample(4));
        }
        else
        {
            TruncatedGaussian measurement_distribution(
                                        true_depth,
                                        sigma_factor_ * true_depth * true_depth,
                                        min_depth_,
                                        max_depth_);

            measured_depth =
                    measurement_distribution.MapStandardGaussian(sample(4));
        }

        return measured_depth;
    }

    virtual Scalar Probability(const Observation& observation) const
    {
        Scalar probability_given_occluded =
                occluded_observation_model_.Probability(observation);

        Scalar probability_given_visible =
                visible_observation_model_.Probability(observation);

        return probability_given_occluded * occlusion_probability_ +
                probability_given_visible * (1.0 - occlusion_probability_);
    }

    virtual Scalar LogProbability(const Observation& observation) const
    {
        return std::log(Probability(observation));
    }

    virtual void Condition(const Scalar& rendered_depth,
                           const Scalar& occlusion)
    {
        rendered_depth_ = rendered_depth;
        occlusion_probability_ = hf::Sigmoid(occlusion);

        occluded_observation_model_.Condition(rendered_depth, true);
        visible_observation_model_.Condition(rendered_depth, false);
    }

private:
    KinectPixelObservationModel occluded_observation_model_;
    KinectPixelObservationModel visible_observation_model_;


    Scalar rendered_depth_;
    Scalar occlusion_probability_;


    TruncatedGaussian object_model_noise_;
    UniformDistribution uniform_distribution_;
    ExponentialDistribution exponential_distribution_;
    UniformDistribution sensor_failure_distribution_;

    // parameters
    const Scalar lambda_, sensor_failure_probability_, object_model_sigma_, sigma_factor_,
            max_depth_, min_depth_;

};

}



#endif
