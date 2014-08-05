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


#ifndef IMAGE_OBSERVATION_MODEL_
#define IMAGE_OBSERVATION_MODEL_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <state_filtering/distributions/distribution.hpp>

namespace distributions
{
template<typename ScalarType_, typename VectorType_, typename MeasurementType_, typename IndexType_ = unsigned>
class RaoBlackwellMeasurementModel: public Distribution<ScalarType_, VectorType_>
{
public:
    typedef typename Distribution<ScalarType_, VectorType_>::ScalarType     ScalarType;
    typedef typename Distribution<ScalarType_, VectorType_>::VectorType     VectorType;
    typedef MeasurementType_                                                MeasurementType;
    typedef IndexType_                                                      IndexType;

public:
    virtual ~RaoBlackwellMeasurementModel() {}

    virtual std::vector<float> Loglikes(const std::vector<VectorType>& states,
                                        std::vector<size_t>& state_indices,
                                        const bool& update = false) = 0;

    virtual void Measurement(const MeasurementType& image,
                             const double& delta_time) = 0;

    // reset the latent variables
    virtual void Reset() = 0;
};

}
#endif
