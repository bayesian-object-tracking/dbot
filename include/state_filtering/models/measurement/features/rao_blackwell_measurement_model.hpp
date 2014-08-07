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


#ifndef MODELS_MEASUREMENT_FEATURES_RAO_BLACKWELL_MEASUREMENT_MODEL_HPP
#define MODELS_MEASUREMENT_FEATURES_RAO_BLACKWELL_MEASUREMENT_MODEL_HPP

#include <vector>
#include <state_filtering/distributions/distribution.hpp>

namespace distributions
{
template<typename ScalarType_, typename StateType_, typename MeasurementType_, typename IndexType_ = size_t>
class RaoBlackwellMeasurementModel: public Distribution<ScalarType_, StateType_>
{
public:
    typedef typename Distribution<ScalarType_, StateType_>::ScalarType     ScalarType;
    typedef typename Distribution<ScalarType_, StateType_>::VectorType     StateType;
    typedef MeasurementType_                                                MeasurementType;
    typedef IndexType_                                                      IndexType;
 public:
    virtual ~RaoBlackwellMeasurementModel() {}

    // since we can not implicitly cast a vector globally we do it here locally
    template<typename Type>
    std::vector<ScalarType> Loglikes(const std::vector<Type>&  states,
                                     std::vector<IndexType>&   indices,
                                     const bool&               update = false)
    {
        std::vector<const StateType*>  state_pointers(states.size());
        for(IndexType i = 0; i < states.size(); i++)
        {
            state_pointers[i] = &(states[i]);
        }

        return Loglikes(state_pointers, indices, update);
    }
    virtual std::vector<ScalarType> Loglikes(const std::vector<const StateType*>&  states,
                                             std::vector<IndexType>&   indices,
                                             const bool&               update = false) = 0;


    virtual void Measurement(const MeasurementType& image, const ScalarType& delta_time) = 0;

    // reset the latent variables
    virtual void Reset() = 0;
};

}
#endif
