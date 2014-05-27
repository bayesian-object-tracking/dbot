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
 * @date 05/25/2014
 * @author Jan Issac (jan.issac@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California (USC),
 *   Karlsruhe Institute of Technology (KIT)
 */

#ifndef STATE_FILTERING_PROCESS_MODEL_COMPOSED_STATIONARY_PROCESS_MODEL_HPP
#define STATE_FILTERING_PROCESS_MODEL_COMPOSED_STATIONARY_PROCESS_MODEL_HPP

// boost
#include <boost/shared_ptr.hpp>

// c++/std
#include <vector>

// eigen
#include <Eigen/Dense>

// state_filtering
#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/gaussian/gaussian_mappable.hpp>
#include <state_filtering/distribution/gaussian/gaussian_sampleable.hpp>
#include <state_filtering/distribution/gaussian/gaussian_distribution.hpp>

#include <state_filtering/process_model/stationary_process_model.hpp>

namespace filter
{

template <>
class ComposedStationaryProcessModel:
        StationaryProcessModel<>
{
public: /* distribution traits */
    typedef StationaryProcessModel<> BaseType;

    enum
    {
        VariableSize = BaseType::VariableSize,
        ControlInputSize = control_input_size,
        RandomSize = random_size
    };
    typedef typename BaseType::ScalarType                           ScalarType;
    typedef typename BaseType::VariableType                         VariableType;
    typedef typename BasenMappableType::RandomType                  RandomType;
    typedef Eigen::Matrix<ScalarType, VariableSize, VariableSize>   CovarianceType;
    typedef Eigen::Matrix<ScalarType, ControlInputSize, 1>          ControlInputType;

    virtual ~ComposedStationaryProcessModel() {}

    virtual VariableType mapFromGaussian(const RandomType& randoms) const
    {
        VariableType variables(variableSize());

        size_t variable_index = 0;
        size_t random_index = 0;
        for(size_t i = 0; i < process_models_.size(); i++)
        {
            variables.middleRows(variable_index, process_models_[i]->variableSize()) =
                    process_models_[i]->mapFromGaussian(randoms.middleRows(random_index, process_models_[i]->count_randoms_));
            variable_index += process_models_[i]->variableSize();
            random_index += process_models_[i]->count_randoms_;
        }
        return variables;
    }

    virtual int variableSize() const
    {

    }

protected:
    const std::vector<boost::shared_ptr<StationaryProcessModel<> > > process_models_;
};

}

#endif
