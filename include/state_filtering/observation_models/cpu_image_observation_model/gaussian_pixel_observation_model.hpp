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


#ifndef GAUSSIAN_OBSERVATION_MODEL_HPP_
#define GAUSSIAN_OBSERVATION_MODEL_HPP_

#include <state_filtering/observation_models/image_observation_model.hpp>

namespace obs_mod
{

class GaussianPixelObservationModel : public PixelObservationModel
{
public:
	GaussianPixelObservationModel(
			float tail_weight = 0.01,
			float model_sigma = 0.003,
			float sigma_factor = 0.00142478,
			float half_life_depth = 1.0,
			float max_depth = 6.0);
	virtual ~GaussianPixelObservationModel();

	virtual float LogProb(float observation, float prediction, bool visible);

	virtual float Prob(float observation, float prediction, bool visible);

private:
	const float exponential_rate_, tail_weight_, model_sigma_, sigma_factor_, max_depth_;
};

}



#endif
