/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
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
 * @date 05/25/2014
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef STATE_FILTERING_PROCESS_MODEL_STATIONARY_PROCESS_MODEL_HPP
#define STATE_FILTERING_PROCESS_MODEL_STATIONARY_PROCESS_MODEL_HPP

// eigen
#include <Eigen/Dense>

// boost
#include <boost/shared_ptr.hpp>

// state_filtering
#include <state_filtering/distribution/features/gaussian_mappable.hpp>

namespace filter
{

template <typename Scalar_ = double,
          int VARIABLE_SIZE = Eigen::Dynamic,
          int CONTROL_SIZE = Eigen::Dynamic,
          int SAMPLE_SIZE = Eigen::Dynamic>
class StationaryProcess:
        public GaussianMappable<Scalar_, VARIABLE_SIZE, SAMPLE_SIZE>
{
public:
    typedef GaussianMappable<Scalar_, VARIABLE_SIZE, SAMPLE_SIZE> Base;

    typedef typename Base::Scalar                               Scalar;
    typedef typename Base::Variable                             Variable;
    typedef typename Base::Sample                               Sample;
    typedef Eigen::Matrix<Scalar, CONTROL_SIZE, 1>              Control;

    virtual ~StationaryProcess() { }

    virtual void conditional(const double& delta_time,
                             const Variable& state,
                             const Control& control) = 0;

    virtual void conditional(const double& delta_time, const Variable& state)
    {
        conditional(delta_time, state, Control::Zero(control_size()));
    }

    virtual int control_size() const = 0;
};

}

#endif
