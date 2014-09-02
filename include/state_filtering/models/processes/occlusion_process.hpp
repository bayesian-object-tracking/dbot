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

#ifndef STATE_FILTERING_OCCLUSION_PROCESS_MODEL_HPP_
#define STATE_FILTERING_OCCLUSION_PROCESS_MODEL_HPP_

#include <state_filtering/models/processes/interfaces/stationary_process_interface.hpp>

// TODO: THIS NEEDS TO BE CLEANED!!
namespace sf
{

// Forward declarations
class OcclusionProcess;

namespace internal
{
/**
 * OcclusionProcess distribution traits specialization
 * \internal
 */
template < >
struct Traits<OcclusionProcess>
{
    enum
    {
        VECTOR_SIZE = 1, //TODO: DO WE NEED THIS?
        DIMENSION = 0 //TODO: THIS IS A HACK
//        CONTROL_SIZE = 0,
//        SAMPLE_SIZE = 0
    };

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, VECTOR_SIZE, 1>   Vector;
    typedef Eigen::Matrix<Scalar, DIMENSION, 1>     Input;

    typedef sf::StationaryProcessInterface<Vector, Input> StationaryProcessInterfaceBase;
};
}

/**
 * \class OcclusionProcess
 *
 * \ingroup distributions
 * \ingroup process_models
 */
class OcclusionProcess:
        public internal::Traits<OcclusionProcess>::StationaryProcessInterfaceBase
{
public:
    typedef internal::Traits<OcclusionProcess> Traits;

    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::Vector State;
    typedef typename Traits::Input  Input;

    enum
    {
        VECTOR_SIZE = Traits::VECTOR_SIZE,
        DIMENSION = Traits::DIMENSION
//        CONTROL_SIZE = Types::CONTROL_SIZE,
//        SAMPLE_SIZE = Types::SAMPLE_SIZE
    };


public:
	// the prob of source being object given source was object one sec ago,
	// and prob of source being object given one sec ago source was not object
    OcclusionProcess(Scalar p_occluded_visible,
                          Scalar p_occluded_occluded)
        : p_occluded_visible_(p_occluded_visible),
          p_occluded_occluded_(p_occluded_occluded),
          c_(p_occluded_occluded_ - p_occluded_visible_),
          log_c_(std::log(c_)) {}

    virtual ~OcclusionProcess() {}


    virtual Scalar Propagate(Scalar occlusion_probability, Scalar delta_time /*in s*/)
    {
        delta_time_ = delta_time;
        occlusion_probability_ = occlusion_probability;
        return MapGaussian()(0);
    }


    virtual void Condition(const Scalar& delta_time,
                              const State& state,
                              const Input& control)
    {
        delta_time_ = delta_time;
        occlusion_probability_ = state(0);
    }

    virtual State MapGaussian() const
    {
        State state_vector;

        if(isnan(delta_time_))
            state_vector(0) =  occlusion_probability_;
        else
        {
            double pow_c_time = std::exp(delta_time_*log_c_);
            state_vector(0) = 1. - (pow_c_time*(1.-occlusion_probability_) + (1 - p_occluded_occluded_)*(pow_c_time-1.)/(c_-1.));
        }
        return state_vector;
    }


    virtual State MapGaussian(const Input& sample) const
    {
        return MapGaussian();
    }

//    virtual int vector_size() const
//    {
//        return VECTOR_SIZE;
//    }
//    virtual int control_size() const
//    {
//        return CONTROL_SIZE;
//    }
//    virtual int NoiseDimension() const
//    {
//        return DIMENSION;
//    }

private:
    // conditionals
    Scalar occlusion_probability_, delta_time_;
    // parameters
    Scalar p_occluded_visible_, p_occluded_occluded_, c_, log_c_;

};

}

#endif
