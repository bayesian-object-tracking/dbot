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


#ifndef MODELS_OBSERVERS_FEATURES_RAO_BLACKWELL_observer_HPP
#define MODELS_OBSERVERS_FEATURES_RAO_BLACKWELL_observer_HPP

#include <vector>
#include <state_filtering/utils/traits.hpp>


namespace sf
{
template<typename State, typename Observation_, typename Index_ = size_t>
class RaoBlackwellObserver
{
public:
    typedef typename internal::VectorTraits<State>::Scalar Scalar;
    typedef Index_        Index;
    typedef Observation_  Observation;

public:
    virtual ~RaoBlackwellObserver() {}

    // since we can not implicitly cast a vector globally we do it here locally
    virtual std::vector<Scalar> Loglikes(const std::vector<State>&  states,
                                     std::vector<Index>&   indices,
                                     const bool&               update = false)
    {
        std::vector<const State*>  state_pointers(states.size());
        for(Index i = 0; i < states.size(); i++)
        {
            state_pointers[i] = &(states[i]);
        }

        return Loglikes_(state_pointers, indices, update);
    }

    /* TODO fix this overloading hack */
    virtual std::vector<Scalar> Loglikes_(const std::vector<const State*>&  states,
                                             std::vector<Index>&   indices,
                                             const bool&               update = false) = 0;


    virtual void SetObservation(const Observation& image, const Scalar& delta_time) = 0;

    // reset the latent variables
    virtual void Reset() = 0;
};

}
#endif
