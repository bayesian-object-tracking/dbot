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

#include <state_filtering/filters/stochastic/coordinate_filter.hpp>

#include <state_filtering/utils/macros.hpp>
#include <state_filtering/utils/helper_functions.hpp>

//#include "image_visualizer.hpp"
#include <omp.h>
#include <string>

#include <boost/lexical_cast.hpp>

using namespace std;
using namespace Eigen;
using namespace distributions;
using namespace boost;

CoordinateParticleFilter::CoordinateParticleFilter(const MeasurementModelPtr observation_model,
                                   const ProcessModelPtr process_model,
                                   const std::vector<std::vector<IndexType> > &independent_blocks,
                                   const double &max_kl_divergence):
    measurement_model_(observation_model),
    process_model_(process_model),
    independent_blocks_(independent_blocks),
    state_distribution_(1), //TODO: THIS IS A HACK, THIS IS NOT GENERAL!!
    max_kl_divergence_(max_kl_divergence)
{
    // make sure sizes are consistent
    dimension_ = 0;
    for(size_t i = 0; i < independent_blocks_.size(); i++)
        for(size_t j = 0; j < independent_blocks_[i].size(); j++)
            dimension_++;

//    if(sample_size != process_model_->NoiseDimension())
//    {
//        cout << "the number of dof in the dependency specification does not correspond to" <<
//                " to the number of dof in the process model!!" << endl;
//        exit(-1);
//    }

//    if(measurement_model_->state_size() != process_model_->variable_size())
//    {
//        cout << "the process and the measurement model do not have the same state size!!" << endl;
//        exit(-1);
//    }
}
CoordinateParticleFilter::~CoordinateParticleFilter() { }






void CoordinateParticleFilter::UpdateWeights(std::vector<float> log_weight_diffs)
{
    for(size_t i = 0; i < log_weight_diffs.size(); i++)
        log_weights_[i] += log_weight_diffs[i];

    vector<float> weights = log_weights_;
    hf::Sort(weights, 1);

    for(int i = weights.size() - 1; i >= 0; i--)
        weights[i] -= weights[0];

    weights = hf::Apply<float, float>(weights, std::exp);
    weights = hf::SetSum(weights, float(1));


    // compute KL divergence to uniform distribution KL(p|u)
    float kl_divergence = std::log(float(weights.size()));
    for(size_t i = 0; i < weights.size(); i++)
    {
        float information = - std::log(weights[i]) * weights[i];
        if(!std::isfinite(information))
            information = 0; // the limit for weight -> 0 is equal to 0
        kl_divergence -= information;
    }

//    cout << "weights " << endl;
//    hf::PrintVector(weights);
    cout << "kl divergence: " << kl_divergence << " max divergence: " << max_kl_divergence_ << endl;

    if(kl_divergence > max_kl_divergence_)
    {
        cout << "resampling!! " << endl;
        std::vector<StateType > new_particles = samples_; // copying stuff around, solved like this because
        // floating body system needs size
        std::vector<double> new_particle_times(samples_.size());
        std::vector<size_t> new_occlusion_indices(samples_.size());

        std::vector<Noise> new_noises(samples_.size());
        std::vector<StateType> new_propagated_particles = samples_; //dito
        std::vector<float> new_loglikes(samples_.size());

        hf::DiscreteSampler sampler(log_weights_); //TESTING

        for(size_t i = 0; i < samples_.size(); i++)
        {
            size_t index = sampler.Sample();
            new_particles[i] = samples_[index];
            new_occlusion_indices[i] = indices_[index];

            new_noises[i] = noises_[index];
            new_propagated_particles[i] = propagated_samples_[index];
            new_loglikes[i] = loglikes_[index];
        }

        samples_          = new_particles;
        indices_  = new_occlusion_indices;
        log_weights_        = std::vector<float>(samples_.size(), 0.);

        noises_ = new_noises;
        propagated_samples_ = new_propagated_particles;
        loglikes_ = new_loglikes;
    }
    else
    {
        cout << "not resampling " << endl;
    }
}











void CoordinateParticleFilter::Filter(const Measurement& measurement,
                                      const ScalarType&  delta_time,
                                      const InputType& input)
{
    INIT_PROFILING;
    measurement_model_->Measurement(measurement, delta_time);

    loglikes_ = std::vector<float>(samples_.size(), 0);
    noises_ = std::vector<Noise>(samples_.size(), Noise::Zero(dimension_));
    propagated_samples_ = samples_;

    for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
    {
        for(size_t particle_index = 0; particle_index < samples_.size(); particle_index++)
        {
            for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                noises_[particle_index](independent_blocks_[block_index][i]) = unit_gaussian_.Sample()(0);

            process_model_->Condition(delta_time,
                                        samples_[particle_index],
                                        input);
            propagated_samples_[particle_index] = process_model_->MapGaussian(noises_[particle_index]);
        }

        RESET;
        bool update_occlusions = (block_index == independent_blocks_.size()-1);
        std::vector<float> new_loglikes = measurement_model_->Loglikes(propagated_samples_,
                                                                       indices_,
                                                                       update_occlusions);
        MEASURE("evaluation ");

        std::vector<float> delta_loglikes(new_loglikes.size());
        for(size_t i = 0; i < delta_loglikes.size(); i++)
            delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
        loglikes_ = new_loglikes;
        UpdateWeights(delta_loglikes);
    }

    samples_ = propagated_samples_;
    state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
}


















void CoordinateParticleFilter::Evaluate(
        const Measurement& observation,
        const double& delta_time,
        const bool& update_occlusions)
{
    // todo at the moment it is assumed that the state times are equal to the observation time
    // we set the observation and evaluate the states ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    measurement_model_->Measurement(observation, delta_time);
    INIT_PROFILING
    family_loglikes_ = measurement_model_->Loglikes(samples_, indices_, update_occlusions);
    MEASURE("observation_model_->Evaluate")
}


void CoordinateParticleFilter::Resample(const int &new_state_count)
{
    size_t state_count;
    if(new_state_count == -1)
        state_count = samples_.size();
    else
        state_count = new_state_count;

    // todo we are not taking multiplicity into account
    std::vector<StateType > new_states(state_count);
    std::vector<double> new_state_times(state_count);
    std::vector<float> new_loglikes(state_count);
    std::vector<float> new_weights(state_count);
    std::vector<size_t> new_occlusion_indices(state_count);

    hf::DiscreteSampler sampler(family_loglikes_);

    for(size_t i = 0; i < state_count; i++)
    {
        size_t index = sampler.Sample();
        new_weights[i] = log_weights_[index];
        new_states[i] = samples_[index];
        new_loglikes[i] = family_loglikes_[index]; // should we set this to 1?
        new_occlusion_indices[i] = indices_[index];
    }

    samples_    		= new_states;
    family_loglikes_= new_loglikes;
    indices_ = new_occlusion_indices;
    log_weights_ = new_weights;




    // TODO: all of this is disgusting. i have to make sure that there is a consistent
    // representation of the hyperstate at all times!
    CoordinateParticleFilter::StateDistributionType::Weights weights(samples_.size());
//    for(size_t i = 0; i < particles_.size(); i++)
//        weights(i) = parent_multiplicities_[i];
//    weights /= weights.sum();

    state_distribution_.SetDeltas(samples_, weights);
}



CoordinateParticleFilter::StateDistributionType &CoordinateParticleFilter::StateDistribution()
{
    return state_distribution_;
}
void CoordinateParticleFilter::Samples(const std::vector<StateType >& states)
{
    // we copy the new states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    samples_ = states;
    indices_ = vector<size_t>(samples_.size(), 0); measurement_model_->Reset();

    family_loglikes_ = std::vector<float>(samples_.size(), 0);
    log_weights_ = std::vector<float>(samples_.size(), 0);
}

const std::vector<CoordinateParticleFilter::StateType>& CoordinateParticleFilter::Samples() const
{
    return samples_;
}



