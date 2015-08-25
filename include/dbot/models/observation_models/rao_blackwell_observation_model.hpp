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

//#include <vector>
#include <Eigen/Core>

#include <fl/util/types.hpp>
//#include <dbot/utils/traits.hpp>




namespace ff
{


/// \todo this observation model is now specific to rigid body rendering,
/// terminology should be adapted accordingly.
template<typename State_, typename Observation_>
class RBObservationModel
{
public:
    typedef State_       State;
    typedef Observation_ Observation;

    typedef Eigen::Array<State, -1, 1>       StateArray;
    typedef Eigen::Array<fl::Real, -1, 1>    RealArray;
    typedef Eigen::Array<int, -1, 1>         IntArray;

public:
    /// constructor and destructor *********************************************
    RBObservationModel(const fl::Real& delta_time): delta_time_(delta_time) { }
    virtual ~RBObservationModel() { }

    /// likelihood computation *************************************************
    virtual RealArray loglikes(const StateArray& deviations,
                               IntArray& indices,
                               const bool& update = false) = 0;

    /// accessors **************************************************************
    virtual void set_observation(const Observation& image) = 0;
    virtual void default_state(const State& state) = 0;
    virtual void reset() = 0;

protected:
    fl::Real delta_time_;
};

}
#endif
