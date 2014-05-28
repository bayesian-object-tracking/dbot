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


#ifndef OCCLUSION_PROCESS_MODEL_HPP_
#define OCCLUSION_PROCESS_MODEL_HPP_

namespace proc_mod
{

class OcclusionProcessModel
{
public:
	// the prob of source being object given source was object one sec ago,
	// and prob of source being object given one sec ago source was not object
	OcclusionProcessModel(double p_visible_visible, double p_visible_occluded);
	virtual ~OcclusionProcessModel();

	// this function returns the prob of source = 1 given the initial
	// prob of source = 1 and the elapsed time.
	virtual double Propagate(double initial_p_source, double time /*in s*/);

private:
	double p_visible_visible_, p_visible_occluded_, log_c_;
};

}



#endif
