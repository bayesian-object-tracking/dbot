/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California,
 *                     Karlsruhe Institute of Technology
 *    Jan Issac (jan.issac@gmail.com)
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @date 05/25/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California (USC),
 *   Karlsruhe Institute of Technology (KIT)
 */

#ifndef STATE_FILTERING_FILTER_PARTICLE_FILTER_CONTEXT_HPP
#define STATE_FILTERING_FILTER_PARTICLE_FILTER_CONTEXT_HPP

#include <limits>
#include <cmath>

#include <Eigen/Dense>

#include <state_filtering/filter/estimate.hpp>
#include <state_filtering/filter/filter_context.hpp>
#include <state_filtering/filter/particle/coordinate_filter.hpp>

namespace filter
{
namespace pfc_internal
{
typedef typename CoordinateFilter::ProcessModel::ControlType ControlType;
// TODO adjust to measurement model
typedef typename CoordinateFilter::MeasurementModel::MeasurementType MeasurementType;
typedef typename FilterContext<ScalarType_, SIZE, pfc_internal::MeasurementType, pfc_internal::ControlType> FilterContext;
}

/**
 * @brief ParticleFilterContext is specialization of @ref filter::FilterContext for particle filters
 */
template <typename ScalarType_, int SIZE>
class ParticleFilterContext: public pfc_internal::FilterContext
{
public:
    typedef typename pfc_internal::FilterContext    EmpiricalMoments;
    typedef typename pfc_internal::MeasurementType  MeasurementType;
    typedef typename pfc_internal::ControlType      ControlType;

    ParticleFilterContext(CoordinateFilter::Ptr filter):
        filter_(filter),
        duration_(0.)
    {

    }

    virtual ~ParticleFilterContext() { }

    /**
     * @brief @ref FilterContext::predictAndUpdate()
     */
    virtual void predictAndUpdate(const MeasurementType& measurement,
                                  double delta_time,
                                  const ControlType& control)
    {
        duration_ += delta_time;

        filter_->Propagate(control, duration_);
        filter_->Evaluate(measurement, duration_, true);
        filter_->Resample();
    }

    /**
     * @return @ref FilterContext::stateDistribution()
     */
    virtual EmpiricalMoments& stateDistribution() const
    {

    }

protected:
    double duration_;
    CoordinateFilter::Ptr filter_;
};

}

#endif
