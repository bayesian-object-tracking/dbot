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
 * @date 2014
 * @author Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
 */

#ifndef FAST_FILTERING_TESTS_DUMMY_MODELS_HPP
#define FAST_FILTERING_TESTS_DUMMY_MODELS_HPP

#include <Eigen/Dense>

#include <fast_filtering/utils/traits.hpp>
#include <fast_filtering/distributions/interfaces/gaussian_map.hpp>
#include <fast_filtering/models/process_models/interfaces/stationary_process_model.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>

template <typename State_> class ProcessModelDummy;

namespace ff
{
namespace internal
{
template <typename State_> struct Traits<ProcessModelDummy<State_> >
{
    typedef State_ State;
    typedef State_ Noise;
    typedef typename State::Scalar Scalar;
    typedef typename StationaryProcessModel<State_>::Input Input;

    typedef StationaryProcessModel<State_> ProcessModelBase;
    typedef GaussianMap<State_, State_> GaussianMapBase;
};
}
}

template <typename State_>
class ProcessModelDummy:
        public ff::internal::Traits<ProcessModelDummy<State_> >::ProcessModelBase,
        public ff::internal::Traits<ProcessModelDummy<State_> >::GaussianMapBase
{
public:
    typedef ff::internal::Traits<ProcessModelDummy<State_> > Traits;

    typedef typename Traits::Scalar Scalar;
    typedef typename Traits::State  State;
    typedef typename Traits::Noise  Noise;
    typedef typename Traits::Input  Input;

    virtual void Condition(const double& delta_time,
                           const State& state,
                           const Input& input)
    {

    }

    virtual State MapStandardGaussian(const Noise& sample) const
    {

    }

    virtual size_t Dimension()
    {
        return State::SizeAtCompileTime;
    }
};

template <typename State>
class ObservationModelDummy
{
public:
    virtual void predict(const State& state)
    {
        // foo
    }

    virtual size_t Dimension() { return 1; }
    virtual size_t NoiseDimension() { return 1; }
};

#endif
