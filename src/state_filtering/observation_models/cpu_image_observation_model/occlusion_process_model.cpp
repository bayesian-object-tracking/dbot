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

#include <state_filtering/observation_models/cpu_image_observation_model/occlusion_process_model.hpp>
#include <math.h>

using namespace proc_mod;

OcclusionProcessModel::OcclusionProcessModel(double p_visible_visible, double p_visible_occluded)
: p_visible_visible_(p_visible_visible), p_visible_occluded_(p_visible_occluded),
log_c_(log(p_visible_visible_ - p_visible_occluded_))
{}

OcclusionProcessModel::~OcclusionProcessModel()
{}

double OcclusionProcessModel::Propagate(double initial_p_source, double time)
{
	if(isnan(time))
		return initial_p_source;
	double c = p_visible_visible_ - p_visible_occluded_;
    double pow_c_time = exp(time*log_c_); // = exp(time*log(p_visible_visible_ - p_visible_occluded_))
	return pow_c_time*initial_p_source + p_visible_occluded_*(pow_c_time-1.)/(c-1.);
}
