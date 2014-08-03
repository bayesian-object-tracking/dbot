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



#include <state_filtering/models/process/stationary_process.hpp>



namespace proc_mod
{

struct OcclusionProcessModelTypes
{
    enum
    {
        VECTOR_SIZE = 1, //TODO: DO WE NEED THIS?
        DIMENSION = 0, //TODO: THIS IS A HACK
//        CONTROL_SIZE = 0,
//        SAMPLE_SIZE = 0
    };

    typedef double ScalarType;
    typedef Eigen::Matrix<ScalarType, VECTOR_SIZE, 1> VectorType;
    typedef Eigen::Matrix<ScalarType, DIMENSION, 1> InputType;

    typedef distributions::StationaryProcess<ScalarType, VectorType, InputType> StationaryProcessType;

};

class OcclusionProcessModel: public OcclusionProcessModelTypes::StationaryProcessType
{
public:
    typedef OcclusionProcessModelTypes::ScalarType ScalarType;
    typedef OcclusionProcessModelTypes::VectorType VectorType;
    typedef OcclusionProcessModelTypes::InputType InputType;


    enum
    {
        VECTOR_SIZE = OcclusionProcessModelTypes::VECTOR_SIZE,
        DIMENSION = OcclusionProcessModelTypes::DIMENSION
//        CONTROL_SIZE = Types::CONTROL_SIZE,
//        SAMPLE_SIZE = Types::SAMPLE_SIZE
    };


public:
	// the prob of source being object given source was object one sec ago,
	// and prob of source being object given one sec ago source was not object
    OcclusionProcessModel(ScalarType p_occluded_visible,
                          ScalarType p_occluded_occluded)
        : p_occluded_visible_(p_occluded_visible),
          p_occluded_occluded_(p_occluded_occluded),
          c_(p_occluded_occluded_ - p_occluded_visible_),
          log_c_(std::log(c_)) {}

    virtual ~OcclusionProcessModel() {}


    virtual ScalarType Propagate(ScalarType occlusion_probability, ScalarType delta_time /*in s*/)
    {
        delta_time_ = delta_time;
        occlusion_probability_ = occlusion_probability;
        return MapGaussian()(0);
    }


    virtual void Condition(const ScalarType& delta_time,
                              const VectorType& state,
                              const InputType& control)
    {
        delta_time_ = delta_time;
        occlusion_probability_ = state(0);
    }

    virtual VectorType MapGaussian() const
    {
        VectorType state_vector;

        if(isnan(delta_time_))
            state_vector(0) =  occlusion_probability_;
        else
        {
            double pow_c_time = std::exp(delta_time_*log_c_);
            state_vector(0) = 1. - (pow_c_time*(1.-occlusion_probability_) + (1 - p_occluded_occluded_)*(pow_c_time-1.)/(c_-1.));
        }
        return state_vector;
    }


    virtual VectorType MapGaussian(const InputType& sample) const
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
    ScalarType occlusion_probability_, delta_time_;
    // parameters
    ScalarType p_occluded_visible_, p_occluded_occluded_, c_, log_c_;

};

}



#endif
