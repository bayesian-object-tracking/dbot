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


#include <state_filtering/observation_models/cpu_image_observation_model/gaussian_pixel_observation_model.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace obs_mod;

GaussianPixelObservationModel::GaussianPixelObservationModel(
		float tail_weight,
		float model_sigma,
		float sigma_factor,
		float half_life_depth,
		float max_depth)
: exponential_rate_(-log(0.5)/half_life_depth),
  tail_weight_(tail_weight),
  model_sigma_(model_sigma),
  sigma_factor_(sigma_factor),
  max_depth_(max_depth)
{}

GaussianPixelObservationModel::~GaussianPixelObservationModel(){}

float GaussianPixelObservationModel::LogProb(float observation, float prediction, bool visible)
{
	return log(Prob(observation, prediction, visible));
}

float GaussianPixelObservationModel::Prob(float observation, float prediction, bool visible)
{
	// todo: if the prediction is infinite, the prob should not depend on visibility. it does not matter
	// for the algorithm right now, but it should be changed

	float sigma = model_sigma_ + sigma_factor_*observation*observation;
	if(visible)
	{
		if(isinf(prediction)) // if the prediction is infinite we return the limit
			return tail_weight_/max_depth_;
		else
			return tail_weight_/max_depth_
					+ (1 - tail_weight_)*exp(-(pow(prediction-observation,2)/(2*sigma*sigma)))
					/ (sqrt(2*M_PI) *sigma);
	}
	else
	{
		if(isinf(prediction)) // if the prediction is infinite we return the limit
			return tail_weight_/max_depth_ +
					(1-tail_weight_)*exponential_rate_*
					exp(0.5*exponential_rate_*(-2*observation + exponential_rate_*sigma*sigma));

		else
			return tail_weight_/max_depth_ +
					(1-tail_weight_)*exponential_rate_*
					exp(0.5*exponential_rate_*(2*prediction-2*observation + exponential_rate_*sigma*sigma))
		*(1+erf((prediction-observation+exponential_rate_*sigma*sigma)/(sqrt(2)*sigma)))
		/(2*(exp(prediction*exponential_rate_)-1));
	}
}
