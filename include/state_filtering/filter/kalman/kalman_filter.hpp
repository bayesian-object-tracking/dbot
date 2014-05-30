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

#ifndef STATE_FILTERING_KALMAN_FILTER_KALMAN_FILTER_HPP
#define STATE_FILTERING_KALMAN_FILTER_KALMAN_FILTER_HPP

// boost
#include <boost/shared_ptr.hpp>

#include <state_filtering/filter/filter_context.hpp>

namespace filter
{

struct Estimate { };
struct Measurement { };

/**
 * @brief The KalmanFilter interface
 */
class KalmanFilter
{
public:
    typedef boost::shared_ptr<KalmanFilter> Ptr;

public:
    virtual ~KalmanFilter() { }

    /**
     * @brief Predicts the state given a delta t and the prior
     *
     * @param [in]  prior_desc        Current prior
     * @param [in]  delta_time        Delta t since the past measurement
     * @param [out] prediction_desc   Predicted state
     */
    virtual void predict(const Estimate& prior_desc,
                         double delta_time,
                         Estimate& prediction_desc);

    /**
     * @brief Updates the prediction given a new measurement
     *
     * @param [in]  measurement       New measurement
     * @param [in]  prediction_desc   Current predicted state
     * @param [out] posterior_desc    State posterior, i.e the updated prediction
     */
    virtual void update(const Measurement& measurement,
                        const Estimate& prediction_desc,
                        Estimate& posterior_desc);
};

}

#endif
