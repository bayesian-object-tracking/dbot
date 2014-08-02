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

#ifndef STATE_FILTERING_DISTRIBUTION_FEATURES_GAUSSIAN_MAPPABLE_HPP
#define STATE_FILTERING_DISTRIBUTION_FEATURES_GAUSSIAN_MAPPABLE_HPP

#include <Eigen/Dense>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <state_filtering/utils/macros.hpp>
#include <state_filtering/distributions/features/sampleable.hpp>

namespace distributions
{

template <typename ScalarType_, typename VectorType_, int INPUT_DIMENSION_EIGEN>
class NormalMappable: public Sampleable<ScalarType_, VectorType_>
{
public:
    typedef typename Sampleable<ScalarType_, VectorType_>::ScalarType       ScalarType;
    typedef typename Sampleable<ScalarType_, VectorType_>::VectorType       VectorType;
    typedef typename Eigen::Matrix<ScalarType, INPUT_DIMENSION_EIGEN, 1>    InputType;

public:
    // constructor and destructor
    NormalMappable(): input_dimension_(INPUT_DIMENSION_EIGEN),
                      generator_(RANDOM_SEED),
                      gaussian_distribution_(0.0, 1.0),
                      gaussian_generator_(generator_, gaussian_distribution_)
    {
        DISABLE_IF_DYNAMIC_SIZE(InputType);
    }
    NormalMappable(const unsigned& input_dimension): input_dimension_(input_dimension),
                                                     generator_(RANDOM_SEED),
                                                     gaussian_distribution_(0.0, 1.0),
                                                     gaussian_generator_(generator_, gaussian_distribution_)
    {
        DISABLE_IF_FIXED_SIZE(InputType);
    }
    virtual ~NormalMappable() { }

    // purely virtual functions
    virtual VectorType MapNormal(const InputType& sample) const = 0;

    // implementations
    virtual int InputDimension() const
    {
        return input_dimension_;
    }
    virtual VectorType Sample()
    {
        InputType normal_sample(InputDimension());
        for (int i = 0; i < InputDimension(); i++)
        {
            normal_sample(i) = gaussian_generator_();
        }
        return MapNormal(normal_sample);
    }

protected:
    const unsigned input_dimension_;

    boost::mt19937 generator_;
    boost::normal_distribution<> gaussian_distribution_;
    boost::variate_generator<boost::mt19937, boost::normal_distribution<> > gaussian_generator_;
};

}

#endif
