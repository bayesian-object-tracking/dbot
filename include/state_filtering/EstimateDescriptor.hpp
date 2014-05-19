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

#ifndef STATE_FILTERING_ESTIMATE_DESCRIPTOR_HPP
#define STATE_FILTERING_ESTIMATE_DESCRIPTOR_HPP

// boost
#include <boost/shared_ptr.hpp>

namespace filter
{
    /**
     * @brief The DynamicVector struct can be anything
     */
    struct DynamicVector {};

    /**
     * @brief The DynamicMatrix struct can be anything
     */
    struct DynamicMatrix {};

    /**
     * @brief The EstimateDescriptor class
     */
    class EstimateDescriptor
    {
    public:
        typedef boost::shared_ptr<EstimateDescriptor> Ptr;

    public:
        virtual ~EstimateDescriptor() { }

        /**
         * @brief Accesses the state as a vector.
         *
         * @return Estimate vector, e.g. mean
         */
        virtual const DynamicVector& estimate() const = 0;

        /**
         * @brief Accesses the covariance of the estimate
         *
         * @return Covariance of the estimate
         */
        virtual const DynamicMatrix& covariance() const = 0;

        /**
         * @brief Returns the time at which the state was updated
         *
         * @return timestamp of update in seconds
         */
        virtual double timestamp() const = 0;

        /**
         * @brief Returns the state dimension
         *
         * @return state dimension
         */
        virtual int dimension() const = 0;

        /**
         * @brief Sets the estimate
         *
         * @param _estimate     The new estimate
         */
        virtual void estimate(const DynamicVector& _estimate) = 0;

        /**
         * @brief Sets the estimate covariance
         *
         * @param _covariance   The new covariance
         */
        virtual void covariance(const DynamicMatrix& covariance) = 0;

        /**
         * @brief Updates the state timestamp
         *
         * @param _time         New update timestamp in seconds
         */
        virtual void timestamp(double _time) = 0;
    };
}

#endif
