/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
 *    Manuel Wuthrich (manuel.wuthrich@gmail.com)
 *    Jan Issac (jan.issac@gmail.com)
 *
 *
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

#ifndef STATE_FILTERING_DISTRIBUTION_INTERFACE_GAUSSIAN_MAPPABLE_INTERFACE_HPP
#define STATE_FILTERING_DISTRIBUTION_INTERFACE_GAUSSIAN_MAPPABLE_INTERFACE_HPP

#include <Eigen/Dense>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <state_filtering/utils/macros.hpp>
#include <state_filtering/utils/traits.hpp>
#include <state_filtering/distributions/interfaces/sampling_interface.hpp>
#include <state_filtering/distributions/standard_gaussian.hpp>

namespace sf
{

template <typename Vector, int NOISE_DIMENSION>
class GaussianMappableInterface:
        public SamplingInterface<Vector>
{
public:
    typedef typename internal::VectorTraits<Vector>::Scalar     Scalar;
    typedef typename Eigen::Matrix<Scalar, NOISE_DIMENSION, 1>  Noise;

public:
    explicit GaussianMappableInterface(const unsigned& noise_dimension = NOISE_DIMENSION):
        standard_gaussian_(noise_dimension)
    {
    }

    virtual ~GaussianMappableInterface() { }

    virtual int NoiseDimension() const
    {
        return standard_gaussian_.Dimension();
    }

    virtual Vector Sample()
    {
        return MapGaussian(standard_gaussian_.Sample());
    }

    virtual Vector MapGaussian(const Noise& sample) const = 0;

private:
    StandardGaussian<Noise> standard_gaussian_;
};

}

#endif
