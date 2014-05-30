/*************************************************************************
This software allows for filtering in high-dimensional measurement and
state spaces, as described in

M. Wuthrich, P. Pastor, M. Kalakrishnan, J. Bohg, and S. Schaal.
Probabilistic Object Tracking using a Range Camera
IEEE/RSJ Intl Conf on Intelligent Robots and Systems, 2013

In a publication based on this software pleace cite the above reference.


Copyright (C) 2014  Manuel Wuthrich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*************************************************************************/


//#define PROFILING_ON

#include <state_filtering/filter/particle/coordinate_filter.hpp>

#include <state_filtering/tools/macros.hpp>
#include <state_filtering/tools/helper_functions.hpp>

//#include "image_visualizer.hpp"
#include <omp.h>
#include <string>

#include <boost/lexical_cast.hpp>

using namespace std;
using namespace Eigen;
using namespace filter;
using namespace boost;

CoordinateFilter::CoordinateFilter(const MeasurementModelPtr observation_model,
                                   const ProcessModelPtr process_model,
                                   const std::vector<std::vector<size_t> >& independent_blocks):
    observation_model_(observation_model),
    process_model_(process_model),
    independent_blocks_(independent_blocks),
    state_distribution_(process_model->variableSize())
{
    // count the degrees of freedom --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    dof_count_ = 0;
    for(size_t i = 0; i < independent_blocks_.size(); i++)
        for(size_t j = 0; j < independent_blocks_[i].size(); j++)
            dof_count_++;
}
CoordinateFilter::~CoordinateFilter() { }

// create offspring
void CoordinateFilter::PartialPropagate(const Eigen::VectorXd& control,
                                        const double& observation_time)
{
    INIT_PROFILING;
    // we write to the following members ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.resize(parents_.size());
    partial_children_.resize(parents_.size());
    partial_children_occlusion_indices_.resize(parents_.size());
    zero_children_.resize(parents_.size());

    for(size_t state_index = 0; state_index < parents_.size(); state_index++)
    {
        // propagation with zero noise is required for normalization ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        process_model_->conditionals(observation_time - parent_times_[state_index], parents_[state_index], control);
        zero_children_[state_index] = process_model_->mapFromGaussian(Eigen::VectorXd::Zero(dof_count_));

        // fill the partial noises and resulting states --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        partial_noises_[state_index].resize(independent_blocks_.size());
        partial_children_[state_index].resize(independent_blocks_.size());
        partial_children_occlusion_indices_[state_index].resize(independent_blocks_.size());
        for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
        {
            partial_noises_[state_index][block_index].resize(parent_multiplicities_[state_index]);
            partial_children_[state_index][block_index].resize(parent_multiplicities_[state_index]);
            partial_children_occlusion_indices_[state_index][block_index].resize(parent_multiplicities_[state_index]);
            for(size_t mult_index = 0; mult_index < parent_multiplicities_[state_index]; mult_index++)
            {
                Eigen::VectorXd partial_noise(Eigen::VectorXd::Zero(dof_count_));
                for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                    partial_noise(independent_blocks_[block_index][i]) = unit_gaussian_.sample()(0);

                partial_noises_[state_index][block_index][mult_index] = partial_noise;
                partial_children_[state_index][block_index][mult_index] = process_model_->mapFromGaussian(partial_noise);
                partial_children_occlusion_indices_[state_index][block_index][mult_index] = parent_occlusion_indices_[state_index];
            }
        }
    }
    MEASURE("creating samples")
}

// evaluate offspring
void CoordinateFilter::PartialEvaluate(const std::vector<float>& observation,
                                       const double& observation_time)
{
    // compute partial_loglikes_ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    observation_model_->set_observation(observation, observation_time);
    hf::Structurer3D converter;
    vector<size_t> flat_partial_children_occlusion_indices_ = converter.Flatten(partial_children_occlusion_indices_);

    INIT_PROFILING;
    partial_children_loglikes_ = converter.Deepen(
                observation_model_->Evaluate(converter.Flatten(partial_children_),
                                             flat_partial_children_occlusion_indices_, false));
    MEASURE("evaluation with " + lexical_cast<string>(flat_partial_children_occlusion_indices_.size()) + " samples");


    vector<float> zero_loglikes = observation_model_->Evaluate(zero_children_, parent_occlusion_indices_, false);
    MEASURE("evaluation with " + lexical_cast<string>(zero_children_.size()) + " samples");

    // compute family_loglikes_ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    family_loglikes_.resize(parents_.size());
    for(size_t state_index = 0; state_index < parents_.size(); state_index++)
    {
        family_loglikes_[state_index] =
                (zero_loglikes[state_index] + log(parent_multiplicities_[state_index])) * (1.0 - float(independent_blocks_.size()));
        for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
        {
            hf::LogSumExp<float> block_loglike;
            for(size_t mult_index = 0; mult_index < parent_multiplicities_[state_index]; mult_index++)
                block_loglike.add_exponent(partial_children_loglikes_[state_index][block_index][mult_index]);
            family_loglikes_[state_index] += block_loglike.Compute();
        }
    }
    MEASURE("computing loglikes_")
}

void CoordinateFilter::PartialResample(const Eigen::VectorXd& control,
                                       const double& observation_time,
                                       const size_t &new_n_states)
{
    // find the children counts per parent ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    INIT_PROFILING;
    hf::DiscreteSampler sampler(family_loglikes_);
    vector<size_t> children_counts(parents_.size(), 0);
    for(size_t new_state_index = 0; new_state_index < new_n_states; new_state_index++)
        children_counts[sampler.sample()]++;
    MEASURE("sampling children_counts");

    // combine partial children into children ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<Eigen::VectorXd> children;
    std::vector<double> children_times;
    vector<size_t> children_multiplicities;
    vector<size_t> children_occlusion_indices;
    for(size_t state_index = 0; state_index < parents_.size(); state_index++)
    {
        if(children_counts[state_index] == 0) continue;
        vector<vector<size_t> >	mult_indices(children_counts[state_index], vector<size_t>(independent_blocks_.size()));
        for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
        {
            hf::DiscreteSampler sampler(partial_children_loglikes_[state_index][block_index]);
            for(size_t child_index = 0; child_index < children_counts[state_index]; child_index++)
                mult_indices[child_index][block_index] = sampler.sample();
        }
        vector<size_t> child_multiplicities;
        hf::SortAndCollapse(mult_indices, child_multiplicities);

        process_model_->conditionals(observation_time - parent_times_[state_index], parents_[state_index], control);
        for(size_t child_index = 0; child_index < mult_indices.size(); child_index++)
        {
            VectorXd noise(VectorXd::Zero(dof_count_));
            for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
                noise += partial_noises_[state_index][block_index][mult_indices[child_index][block_index]];

            children.push_back(process_model_->mapFromGaussian(noise));
            children_times.push_back(observation_time);
            children_multiplicities.push_back(child_multiplicities[child_index]);
            children_occlusion_indices.push_back(parent_occlusion_indices_[state_index]);
        }
    }
    MEASURE("sampling children");

    // children become new parents ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parents_ = children;
    parent_times_ = children_times;
    parent_multiplicities_ = children_multiplicities;
    parent_occlusion_indices_ = children_occlusion_indices;
    family_loglikes_ = vector<float>(parents_.size(), 0);

    // new parents have no children yet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.clear();
    partial_children_.clear();
    partial_children_occlusion_indices_.clear();
    zero_children_.clear();
    partial_children_loglikes_.clear();
}


void CoordinateFilter::UpdateOcclusions(const std::vector<float>& observation,
                                        const double& observation_time)
{
    observation_model_->set_observation(observation, observation_time);
    INIT_PROFILING;
    observation_model_->Evaluate(parents_, parent_occlusion_indices_, true);
    MEASURE("evaluation with " + lexical_cast<string>(parents_.size()) + " samples")

}





void CoordinateFilter::Enchilada(
        const Eigen::VectorXd control,
        const double &observation_time,
        const std::vector<float>& observation,
        const size_t &new_n_states)
{
    PartialPropagate(control, observation_time);
    PartialEvaluate(observation, observation_time);
    PartialResample(control, observation_time, new_n_states);
    UpdateOcclusions(observation, observation_time);
}



void CoordinateFilter::Propagate(
        const Eigen::VectorXd control,
        const double &current_time)
{
    // we propagate the states to the current time, appying the control input ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for(size_t state_index = 0; state_index < parents_.size(); state_index++)
    {        
        process_model_->conditionals(current_time - parent_times_[state_index], parents_[state_index], control);

        Eigen::VectorXd iso_sample(dof_count_);
        for (int i = 0; i < iso_sample.rows(); i++)
            iso_sample(i) = unit_gaussian_.sample()(0);

        parents_[state_index] = process_model_->mapFromGaussian(iso_sample);

        parent_times_[state_index] = current_time;
    }
}


void CoordinateFilter::Evaluate(
        const std::vector<float>& observation,
        const double& observation_time,
        const bool& update_occlusions)
{
    // todo at the moment it is assumed that the state times are equal to the observation time
    // we set the observation and evaluate the states ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    observation_model_->set_observation(observation, observation_time);
    INIT_PROFILING
    family_loglikes_ = observation_model_->Evaluate(parents_, parent_occlusion_indices_, update_occlusions);
    MEASURE("observation_model_->Evaluate")
}


void CoordinateFilter::Evaluate_test(
        const std::vector<float>& observation,
        const double& observation_time,
        const bool& update_occlusions,
        vector<vector<int> > intersect_indices,
        vector<vector<float> > predictions)
{
    observation_model_->set_observation(observation, observation_time);
    family_loglikes_ = observation_model_->Evaluate_test(parents_, parent_occlusion_indices_, update_occlusions, intersect_indices, predictions);
}


void CoordinateFilter::Resample(const int &new_state_count)
{
    size_t state_count;
    if(new_state_count == -1)
        state_count = parents_.size();
    else
        state_count = new_state_count;

    // todo we are not taking multiplicity into account
    std::vector<Eigen::VectorXd > new_states(state_count);
    std::vector<double> new_state_times(state_count);
    std::vector<size_t> new_multiplicities(state_count);
    std::vector<float> new_loglikes(state_count);
    std::vector<size_t> new_occlusion_indices(state_count);

    hf::DiscreteSampler sampler(family_loglikes_);

    for(size_t i = 0; i < state_count; i++)
    {
        size_t index = sampler.sample();
        new_states[i] = parents_[index];
        new_state_times[i] = parent_times_[index];
        new_multiplicities[i] = parent_multiplicities_[index];
        new_loglikes[i] = family_loglikes_[index]; // should we set this to 1?
        new_occlusion_indices[i] = parent_occlusion_indices_[index];
    }

    parents_    		= new_states;
    parent_times_    = new_state_times;
    parent_multiplicities_ = new_multiplicities;
    family_loglikes_= new_loglikes;
    parent_occlusion_indices_ = new_occlusion_indices;




    // TODO: all of this is disgusting. i have to make sure that there is a consistent
    // representation of the hyperstate at all times!
    CoordinateFilter::StateDistribution::Weights weights(parents_.size());
    for(size_t i = 0; i < parents_.size(); i++)
        weights(i) = parent_multiplicities_[i];
    weights /= weights.sum();

    state_distribution_.setDeltas(parents_, weights);
}


void CoordinateFilter::Sort()
{
    vector<int> indices = hf::SortDescend(family_loglikes_);

    std::vector<Eigen::VectorXd > sorted_states(indices.size());
    std::vector<double> sorted_state_times(indices.size());
    std::vector<size_t> sorted_multiplicities(indices.size());
    std::vector<float> sorted_loglikes(indices.size());
    std::vector<size_t> sorted_occlusion_indices(indices.size());
    for(size_t i = 0; i < indices.size(); i++)
    {
        sorted_states[i] = parents_[indices[i]];
        sorted_state_times[i] = parent_times_[indices[i]];
        sorted_multiplicities[i] = parent_multiplicities_[indices[i]];
        sorted_loglikes[i] = family_loglikes_[indices[i]]; // should we set this to 1?
        sorted_occlusion_indices[i] = parent_occlusion_indices_[indices[i]];
    }

    parents_    		= sorted_states;
    parent_times_    = sorted_state_times;
    parent_multiplicities_ = sorted_multiplicities;
    family_loglikes_= sorted_loglikes;
    parent_occlusion_indices_ = sorted_occlusion_indices;
}




CoordinateFilter::StateDistribution &CoordinateFilter::stateDistribution()
{
    return state_distribution_;
}



// set and get fcts ==========================================================================================================================================================================================================================================================================================================================================================================================
void CoordinateFilter::get(MeasurementModelPtr &observation_model) const
{
    observation_model = observation_model_;
}
void CoordinateFilter::get(ProcessModelPtr &process_model) const
{
    process_model = process_model_;
}
void CoordinateFilter::get(std::vector<Eigen::VectorXd >& states) const
{
    states = parents_;
}
void CoordinateFilter::get(std::vector<double>& state_times) const
{
    state_times = parent_times_;
}
void CoordinateFilter::get(std::vector<size_t>& multiplicities) const
{
    multiplicities = parent_multiplicities_;
}
void CoordinateFilter::get(std::vector<float>& loglikes) const
{
    loglikes = family_loglikes_;
}
const Eigen::VectorXd& CoordinateFilter::get_state(size_t index) const
{
    return parents_[index];
}
const std::vector<float> CoordinateFilter::get_occlusions(size_t index) const
{
    return observation_model_->get_occlusions(parent_occlusion_indices_[index]	);
}
void CoordinateFilter::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                             std::vector<std::vector<float> > &depth)
{
    observation_model_->get_depth_values(intersect_indices, depth);
}


void CoordinateFilter::set_states(
        const std::vector<Eigen::VectorXd >& states,
        const std::vector<double>& state_times,
        const std::vector<size_t>& multiplicities,
        const std::vector<float>& loglikes)
{
    // we copy the new states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    parents_ = states;
    parent_times_ = state_times;
    parent_multiplicities_ = multiplicities;
    parent_occlusion_indices_ = vector<size_t>(parents_.size(), 0); observation_model_->set_occlusions();
    family_loglikes_ = loglikes;

    // if some arguments have not been passed we set to default values ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(parent_times_.empty()) parent_times_ = std::vector<double>(parents_.size(), 0);
    if(parent_multiplicities_.empty()) parent_multiplicities_ = std::vector<size_t>(parents_.size(), 1);
    if(family_loglikes_.empty()) family_loglikes_ = std::vector<float>(parents_.size(), 0);

    // new parents have no children yet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.clear();
    partial_children_.clear();
    partial_children_occlusion_indices_.clear();
    zero_children_.clear();
    partial_children_loglikes_.clear();


}
void CoordinateFilter::set_independence(const std::vector<std::vector<size_t> >& independent_blocks)
{
    independent_blocks_ = independent_blocks;
}
void CoordinateFilter::set(const MeasurementModelPtr &observation_model)
{
    observation_model_ = observation_model;
}
void CoordinateFilter::set(const ProcessModelPtr &process_model)
{
    process_model_ = process_model;
}


