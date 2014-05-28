/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Max-Planck-Institute for Intelligent Systems,
 *                     University of Southern California
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
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef STATE_FILTERING_DISTRIBUTION_DISTRIBUTION_HPP
#define STATE_FILTERING_DISTRIBUTION_DISTRIBUTION_HPP

// eigen
#include <Eigen/Dense>

/*
 * Enable and disable functions macros for fixed and dynamic sized distributions
 */
#define DISABLE_CONSTRUCTOR_IF_DYNAMIC_SIZE(VariableType) \
            if (filter::internals::Invalidate<VariableType::SizeAtCompileTime != Eigen::Dynamic> \
                ::YOU_CALLED_A_FIXED_SIZE_CONSTRUCTOR_ON_A_DYNAMIC_SIZE_DISTRIBUTION) { }

#define DISABLE_CONSTRUCTOR_IF_FIXED_SIZE(VariableType) \
    if (filter::internals::Invalidate<VariableType::SizeAtCompileTime == Eigen::Dynamic> \
        ::YOU_CALLED_A_DYNAMIC_SIZE_CONSTRUCTOR_ON_A_FIXED_SIZE_DISTRIBUTION) { }


namespace filter
{
namespace internals
{
template <bool Condition> struct Invalidate { };

template <>
struct Invalidate<true>
{
    enum
    {
        YOU_CALLED_A_FIXED_SIZE_CONSTRUCTOR_ON_A_DYNAMIC_SIZE_DISTRIBUTION,
        YOU_CALLED_A_DYNAMIC_SIZE_CONSTRUCTOR_ON_A_FIXED_SIZE_DISTRIBUTION
    };
};
}

template <typename ScalarType_, int VariableSize>
class Distribution
{
public:    
    typedef ScalarType_                         ScalarType;
    typedef Eigen::Matrix<ScalarType, VariableSize, 1>  VariableType;


    /**
     * @brief Overridable virtual destructor
     */
    virtual ~Distribution() { }

    /**
     * @brief Returns current variable VariableSize ()
     *
     * @return variable VariableSize for dynamic and fixed VariableSize (dimensional) distributions
     */
    virtual int variableSize() const = 0;
};

}

#endif
