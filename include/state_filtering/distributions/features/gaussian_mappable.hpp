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
#include <state_filtering/utils/traits.hpp>
#include <state_filtering/distributions/features/sampleable.hpp>

namespace sf
{

template <typename Vector, int NOISE_DIMENSION>
class GaussianMappable:
        public Sampleable<Vector>
{
public:
    typedef typename internal::VectorTraits<Vector>::Scalar     Scalar;
    typedef typename Eigen::Matrix<Scalar, NOISE_DIMENSION, 1>  Noise;

public:
    // constructor and destructor
    GaussianMappable(): noise_dimension_(NOISE_DIMENSION),
                        generator_(RANDOM_SEED),
                        gaussian_distribution_(0.0, 1.0),
                        gaussian_generator_(generator_, gaussian_distribution_)
    {
        SF_DISABLE_IF_DYNAMIC_SIZE(Noise);
    }
    GaussianMappable(const unsigned& noise_dimension): noise_dimension_(noise_dimension),
                                                       generator_(RANDOM_SEED),
                                                       gaussian_distribution_(0.0, 1.0),
                                                       gaussian_generator_(generator_, gaussian_distribution_)
    {
        SF_DISABLE_IF_FIXED_SIZE(Noise);
    }
    virtual ~GaussianMappable() { }

    // purely virtual functions
    virtual Vector MapGaussian(const Noise& sample) const = 0;

    // implementations
    virtual int NoiseDimension() const
    {
        return noise_dimension_;
    }
    virtual Vector Sample()
    {
        Noise gaussian_sample(NoiseDimension());
        for (int i = 0; i < NoiseDimension(); i++)
        {
            gaussian_sample(i) = gaussian_generator_();
        }
        return MapGaussian(gaussian_sample);
    }

private:
    unsigned noise_dimension_;

    boost::mt19937 generator_;
    boost::normal_distribution<> gaussian_distribution_;
    boost::variate_generator<boost::mt19937, boost::normal_distribution<> > gaussian_generator_;
};

}

#endif
