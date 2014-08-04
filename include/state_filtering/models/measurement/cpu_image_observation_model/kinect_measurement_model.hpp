/*************************************************************************
This software allows for filtering in high-dimensional measurement and
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


#ifndef KINECT_MEASUREMENT_MODEL_HPP_
#define KINECT_MEASUREMENT_MODEL_HPP_

#include <state_filtering/distributions/features/evaluable.hpp>
#include <cmath>

#include <iostream>

namespace obs_mod
{

class KinectMeasurementModel: public distributions::Evaluable<double, double>
{
public:
    typedef distributions::Evaluable<double, double >   BaseType;
    typedef typename BaseType::ScalarType               ScalarType;
    typedef typename BaseType::VectorType               VectorType;


    KinectMeasurementModel(ScalarType tail_weight = 0.01,
                           ScalarType model_sigma = 0.003,
                           ScalarType sigma_factor = 0.00142478,
                           ScalarType half_life_depth = 1.0,
                           ScalarType max_depth = 6.0)
        : exponential_rate_(-log(0.5)/half_life_depth),
          tail_weight_(tail_weight),
          model_sigma_(model_sigma),
          sigma_factor_(sigma_factor),
          max_depth_(max_depth) { }

    virtual ~KinectMeasurementModel() {}

    virtual ScalarType Probability(const VectorType& measurement) const
    {
        // todo: if the prediction is infinite, the prob should not depend on visibility. it does not matter
        // for the algorithm right now, but it should be changed
        ScalarType probability;
        ScalarType sigma = model_sigma_ + sigma_factor_*measurement*measurement;
        if(!occlusion_)
        {
            if(isinf(prediction_)) // if the prediction_ is infinite we return the limit
                probability = tail_weight_/max_depth_;
            else
                probability = tail_weight_/max_depth_
                        + (1 - tail_weight_)*std::exp(-(pow(prediction_-measurement,2)/(2*sigma*sigma)))
                        / (sqrt(2*M_PI) *sigma);
        }
        else
        {
            if(isinf(prediction_)) // if the prediction_ is infinite we return the limit
                probability = tail_weight_/max_depth_ +
                        (1-tail_weight_)*exponential_rate_*
                        std::exp(0.5*exponential_rate_*(-2*measurement + exponential_rate_*sigma*sigma));

            else
                probability = tail_weight_/max_depth_ +
                        (1-tail_weight_)*exponential_rate_*
                        std::exp(0.5*exponential_rate_*(2*prediction_-2*measurement + exponential_rate_*sigma*sigma))
            *(1+erf((prediction_-measurement+exponential_rate_*sigma*sigma)/(sqrt(2)*sigma)))
            /(2*(std::exp(prediction_*exponential_rate_)-1));
        }

        return probability;
    }

    virtual ScalarType LogProbability(const VectorType& measurement) const
    {
        return std::log(Probability(measurement));
    }

    virtual void Condition(const ScalarType& prediction, const bool& occlusion)
    {
        prediction_ = prediction;
        occlusion_ = occlusion;
    }

private:
    const ScalarType exponential_rate_, tail_weight_, model_sigma_, sigma_factor_, max_depth_;

    ScalarType prediction_;
    bool occlusion_;
};

}



#endif
