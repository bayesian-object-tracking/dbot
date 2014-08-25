/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
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
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems,
 *  University of Southern California
 */

#ifndef STATE_FILTERING_UTILS_TRAITS_HPP
#define STATE_FILTERING_UTILS_TRAITS_HPP

#include <complex>
#include <Eigen/Dense>

namespace sf
{

namespace internal
{
/**
 * \internal
 * Generic distribution trait template
 */
template <typename T> struct Traits { };

/**
 * \internal
 * Generic Vector Trait, it assumes that is derived from an Eigen::Matrix having
 * a Scalar type
 */
template <typename Vector>
struct VectorTraits
{
    typedef typename Vector::Scalar Scalar;
    enum { Dimension = Vector::SizeAtCompileTime };
};

// VectorTraits specializations for single dimension vectors (scalars of a
// specific type)

template <>
struct VectorTraits<int> { typedef int Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<float> { typedef float Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<double> { typedef double Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<long double> { typedef long double Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<std::complex<float> > { typedef std::complex<float> Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<std::complex<double> > { typedef std::complex<double> Scalar; enum { Dimension = 1 }; };

template <>
struct VectorTraits<std::complex<long double> > { typedef std::complex<long double> Scalar; enum { Dimension = 1 }; };

}

}

#endif

