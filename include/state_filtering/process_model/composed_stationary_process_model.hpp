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
 *    Jan Issac (jan.issac@gmail.com)
 * @author Manuel Wuthrich (manuel.wuthrich@gmail.com)
 * Max-Planck-Institute for Intelligent Systems, University of Southern California
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
#include <state_filtering/distribution/features/evaluable.hpp>
#include <state_filtering/distribution/features/gaussian_mappable.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>

#include <state_filtering/process_model/stationary_process_model.hpp>

namespace filter
{

class ComposedStationaryProcessModel:
        public StationaryProcess<>
{
public: /* distribution traits */
    typedef StationaryProcess<> Base;

    typedef typename Base::Scalar       Scalar;
    typedef typename Base::Variable     Variable;
    typedef typename Base::Sample      Sample;
    typedef typename Base::Control Control;

    typedef boost::shared_ptr<Base> StationaryProcessModelPtr;
    typedef std::vector<StationaryProcessModelPtr> ProcessModelList;

    using Base::conditional;

    ComposedStationaryProcessModel(ProcessModelList process_models):
        process_models_(process_models)
    {
    }

    virtual ~ComposedStationaryProcessModel() {}

    virtual Variable mapNormal(const Sample& randoms) const
    {
        Variable variables(variable_size());

        size_t variable_index = 0;
        size_t random_index = 0;
        for(size_t i = 0; i < process_models_.size(); i++)
        {
            variables.middleRows(variable_index, process_models_[i]->variable_size()) =
                    process_models_[i]->mapNormal(randoms.middleRows(random_index, process_models_[i]->sample_size()));
            variable_index += process_models_[i]->variable_size();
            random_index += process_models_[i]->sample_size();
        }
        return variables;
    }

    // set functions with arguments specific to stationary process models ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    virtual void conditional(
            const double& delta_time,
            const Variable& state,
            const Control& control)
    {
        size_t state_index = 0;
        size_t control_index = 0;
        for(size_t i = 0; i < process_models_.size(); i++)
        {
            process_models_[i]->conditional(
                        delta_time,
                        state.middleRows(state_index, process_models_[i]->variable_size()),
                        control.middleRows(control_index, process_models_[i]->control_size()));
            state_index += process_models_[i]->variable_size();
            control_index += process_models_[i]->control_size();
        }
    }

    virtual int variable_size() const
    {
        return total_count_state(process_models_);
    }

    virtual int sample_size() const
    {
        return total_count_randoms(process_models_);
    }

    virtual int control_size() const
    {
        return total_count_control(process_models_);
    }

protected:
    const ProcessModelList process_models_;

private:
    // silly counting functions ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    static unsigned total_count_state(std::vector<boost::shared_ptr<StationaryProcess<> > > process_models)
    {
        unsigned total_count_state = 0;
        for(size_t i = 0; i < process_models.size(); i++)
            total_count_state += process_models[i]->variable_size();
        return total_count_state;
    }
    static unsigned total_count_control(std::vector<boost::shared_ptr<StationaryProcess<> > > process_models)
    {
        unsigned total_count_control = 0;
        for(size_t i = 0; i < process_models.size(); i++)
            total_count_control += process_models[i]->control_size();
        return total_count_control;
    }

    static unsigned total_count_randoms(std::vector<boost::shared_ptr<StationaryProcess<> > > process_models)
    {
        unsigned total_count_randoms = 0;
        for(size_t i = 0; i < process_models.size(); i++)
            total_count_randoms += process_models[i]->sample_size();
        return total_count_randoms;
    }
};

}

#endif
