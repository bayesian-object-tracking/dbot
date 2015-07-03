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


#ifndef FAST_FILTERING_MODELS_OBSERVATION_MODELS_INTERFACES_RAO_BLACKWELL_OBSERVATION_MODEL_HPP
#define FAST_FILTERING_MODELS_OBSERVATION_MODELS_INTERFACES_RAO_BLACKWELL_OBSERVATION_MODEL_HPP

#include <vector>
#include <dbot/utils/traits.hpp>


namespace ff
{

/**
 * Rao-Blackwellized particle filter observation model interface
 *
 * \ingroup observation_models
 */
template<typename State_, typename Observation_>
class RaoBlackwellObservationModel
{
public:
    typedef State_       State;
    typedef Observation_ Observation;

public:
    RaoBlackwellObservationModel(const double& delta_time):
    delta_time_(delta_time)
    {}

    virtual ~RaoBlackwellObservationModel() { }

    virtual std::vector<double> Loglikes(const std::vector<State>& states,
                                         std::vector<size_t>& indices,
                                         const bool& update = false) = 0;

    virtual void SetObservation(const Observation& image) = 0;

    // reset the latent variables
    virtual void Reset() = 0;


protected:
    double delta_time_;
};

}
#endif
