/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California,
 *                     Karlsruhe Institute of Technology
 *    Jan Issac (jan.issac@gmail.com)
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
 * @date 05/19/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California (USC),
 *   Karlsruhe Institute of Technology (KIT)
 */

#ifndef STATE_FILTERING_FILTER_FILTER_CONTEXT_HPP
#define STATE_FILTERING_FILTER_FILTER_CONTEXT_HPP

#include <state_filtering/filter/types.hpp>
#include <state_filtering/distribution/empirical_moments.hpp>

/**
 * @brief State filtering namespace
 */
namespace filter
{

/**
 * @brief FilterContext is a generic interface of a context containing a filter algorithm
 *
 * FilterContext is a generic interface of a filter algorithm context. A specialization of this
 * may run any type of filter with its own interface. Each filter must be able to provide at
 * least the three functions this interface provides. The underlying algorithm and may use any
 * representation for its data and may have any interface it requires.
 *
 * The Filter context layer provides a simple and minimal interface for the client. It also
 * provides the possibility to use stateless and stateful filters. Stateless filters are a clean
 * way to implement an algorithm, however the client has to maintain the state. In this case
 * FilterContext takes care of this matter. A stateful filter is a filter which stores and
 * manages the state internally. This may due to performance reasons such as in the case of
 * GPU implementations. Both, stateless and stateful filter can be used in the same fashion.
 */
template <typename ScalarType_, int SIZE, typename MeasurementType, typename ControlType>
class FilterContext
{
public:
    typedef EmpiricalMoments<ScalarType_, SIZE> EmpiricalMoments;

    virtual ~FilterContext() { }

    /**
     * @brief Propagates the current state and updates it using the measurement
     *
     * @param measurement   Most recent measurement used to update the state
     * @param time          Current timestamp
     */
    virtual void predictAndUpdate(const MeasurementType& measurement,
                                  double time,
                                  const ControlType& control) = 0;

    /**
     * @return Accesses the filtered state
     */
    virtual EmpiricalMoments& stateDistribution() const = 0;
};

}

#endif
