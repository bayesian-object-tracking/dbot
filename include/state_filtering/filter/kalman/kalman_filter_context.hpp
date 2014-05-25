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

#ifndef STATE_FILTERING_KALMAN_FILTER_KALMAN_FILTER_CONTEXT_HPP
#define STATE_FILTERING_KALMAN_FILTER_KALMAN_FILTER_CONTEXT_HPP

#include <state_filtering/filter/estimate.hpp>
#include <state_filtering/filter/filter_context.hpp>
#include <state_filtering/filter/kalman/kalman_filter.hpp>

namespace filter
{

/**
 * @brief KalmanFilterContext is specialization of @ref filter::FilterContext for Kalman filters
 */
class KalmanFilterContext:
        public FilterContext<KalmanFilter::Ptr>
{
public:
    virtual ~KalmanFilterContext() { }

    /**
     * Propagates the current state and updates it using the measurement
     *
     * @param [in] measurement  Most recent measurement used to update the state
     */
    virtual void propagateAndUpdate(const Measurement& measurement);

    /**
     * @return Copy of the current state.
     */
    virtual const EstimateDescriptor::Ptr state();

    /**
     * @return Accesses the filter algorithm
     */
    virtual KalmanFilter::Ptr filter();

protected:
    KalmanFilter::Ptr kalman_filter_;
};

}

#endif
