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

#ifndef POSE_TRACKING_MODELS_PROCESS_MODELS_OCCLUSION_PROCESS_MODEL_HPP
#define POSE_TRACKING_MODELS_PROCESS_MODELS_OCCLUSION_PROCESS_MODEL_HPP

#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>
#include <fast_filtering/distributions/interfaces/gaussian_map.hpp>

// TODO: THIS IS JUST A LINEAR GAUSSIAN PROCESS WITH NO NOISE, SHOULD DISAPPEAR
namespace ff
{

/**
 * \class OcclusionProcess
 *
 * \ingroup distributions
 * \ingroup process_models
 */
class OcclusionProcessModel:
        public StationaryProcessModel<double>,
        public GaussianMap<double>
{
public:
	// the prob of source being object given source was object one sec ago,
	// and prob of source being object given one sec ago source was not object
    OcclusionProcessModel(double p_occluded_visible,
                          double p_occluded_occluded):
                                      p_occluded_visible_(p_occluded_visible),
                                      p_occluded_occluded_(p_occluded_occluded),
                                      c_(p_occluded_occluded_ - p_occluded_visible_),
                                      log_c_(std::log(c_)) { }

    virtual ~OcclusionProcessModel() {}

    virtual void Condition(const double& delta_time,
                           const double& occlusion_probability,
                           const StationaryProcessModel<double>::Input& input = StationaryProcessModel<double>::Input())
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

#endif
