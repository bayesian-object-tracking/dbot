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
                                   const std::vector<std::vector<size_t> >& independent_blocks,
                                   const double &max_kl_divergence):
    measurement_model_(observation_model),
    process_model_(process_model),
    independent_blocks_(independent_blocks),
    state_distribution_(process_model->InputDimension() * 2), //TODO: THIS IS A HACK, THIS IS NOT GENERAL!!
    max_kl_divergence_(max_kl_divergence)
{
    // make sure sizes are consistent
    size_t sample_size = 0;
    for(size_t i = 0; i < independent_blocks_.size(); i++)
        for(size_t j = 0; j < independent_blocks_[i].size(); j++)
            sample_size++;

    if(sample_size != process_model_->InputDimension())
    {
        cout << "the number of dof in the dependency specification does not correspond to" <<
                " to the number of dof in the process model!!" << endl;
        exit(-1);
    }

//    if(measurement_model_->state_size() != process_model_->variable_size())
//    {
//        cout << "the process and the measurement model do not have the same state size!!" << endl;
//        exit(-1);
//    }
}
CoordinateParticleFilter::~CoordinateParticleFilter() { }




void CoordinateParticleFilter::Enchilada(
        const Control control,
        const double &observation_time,
        const Measurement& observation,
        const size_t &new_n_states)
{
    cout << "DONT USE THIS FUNCTION ANYMORE!!" <<  endl;
    exit(-1);
}






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
        std::vector<State > new_particles(particles_.size());
        std::vector<double> new_particle_times(particles_.size());
        std::vector<size_t> new_occlusion_indices(particles_.size());

        std::vector<Noise> new_noises(particles_.size());
        std::vector<State> new_propagated_particles(particles_.size());
        std::vector<float> new_loglikes(particles_.size());

        hf::DiscreteSampler sampler(log_weights_); //TESTING

        for(size_t i = 0; i < particles_.size(); i++)
        {
            size_t index = sampler.Sample();
            new_particles[i] = particles_[index];
            new_particle_times[i] = particle_times_[index];
            new_occlusion_indices[i] = occlusion_indices_[index];

            new_noises[i] = noises_[index];
            new_propagated_particles[i] = propagated_particles_[index];
            new_loglikes[i] = loglikes_[index];
        }

        particles_          = new_particles;
        particle_times_     = new_particle_times;
        occlusion_indices_  = new_occlusion_indices;
        log_weights_        = std::vector<float>(particles_.size(), 0.);

        noises_ = new_noises;
        propagated_particles_ = new_propagated_particles;
        loglikes_ = new_loglikes;
    }
    else
    {
        cout << "not resampling " << endl;
    }
}











void CoordinateParticleFilter::Filter( const Control control,
                                       const double &observation_time,
                                       const Measurement& observation)
{
    INIT_PROFILING;
    measurement_model_->measurement(observation, observation_time);

    loglikes_ = std::vector<float>(particles_.size(), 0);
    noises_ = std::vector<Noise>(particles_.size(), Noise::Zero(process_model_->InputDimension()));
    propagated_particles_ = std::vector<State>(particles_.size());

    for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
    {
        for(size_t particle_index = 0; particle_index < particles_.size(); particle_index++)
        {
            for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                noises_[particle_index](independent_blocks_[block_index][i]) = unit_gaussian_.Sample()(0);

            process_model_->Conditional(observation_time - particle_times_[particle_index],
                                        particles_[particle_index],
                                        control);
            propagated_particles_[particle_index] = process_model_->MapGaussian(noises_[particle_index]);
        }

        RESET;
        bool update_occlusions = (block_index == independent_blocks_.size()-1);
        std::vector<float> new_loglikes = measurement_model_->Evaluate(propagated_particles_,
                                                                       occlusion_indices_,
                                                                       update_occlusions);
        MEASURE("evaluation ");
        cout << "evaluated block " << block_index + 1 << " of " << independent_blocks_.size() << endl
             << " with " << particles_.size() << " particles." << endl
             << "updating occlusion: " << update_occlusions << endl;

        std::vector<float> delta_loglikes(new_loglikes.size());
        for(size_t i = 0; i < delta_loglikes.size(); i++)
            delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
        loglikes_ = new_loglikes;
        UpdateWeights(delta_loglikes);
    }

    particles_ = propagated_particles_;
    for(size_t i = 0; i < particle_times_.size(); i++)
        particle_times_[i] = observation_time;

    state_distribution_.SetDeltas(particles_); // not sure whether this is the right place
}








void CoordinateParticleFilter::set_states(const std::vector<State >& states,
                                  const std::vector<double>& state_times,
                                  const std::vector<float> &loglikes)
{
    // we copy the new states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    particles_ = states;
    particle_times_ = state_times;
    occlusion_indices_ = vector<size_t>(particles_.size(), 0); measurement_model_->set_occlusions();
    family_loglikes_ = loglikes;

    // if some arguments have not been passed we set to default values ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(particle_times_.empty()) particle_times_ = std::vector<double>(particles_.size(), 0);
    if(family_loglikes_.empty()) family_loglikes_ = std::vector<float>(particles_.size(), 0);
    log_weights_ = std::vector<float>(particles_.size(), 0);
}





































void CoordinateParticleFilter::Propagate(
        const Control control,
        const double &current_time)
{
    // we propagate the states to the current time, appying the control input ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
    {        
        process_model_->Conditional(current_time - particle_times_[state_index], particles_[state_index], control);
        particles_[state_index] = process_model_->Sample();
        particle_times_[state_index] = current_time;
    }
}


void CoordinateParticleFilter::Evaluate(
        const Measurement& observation,
        const double& observation_time,
        const bool& update_occlusions)
{
    // todo at the moment it is assumed that the state times are equal to the observation time
    // we set the observation and evaluate the states ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    measurement_model_->measurement(observation, observation_time);
    INIT_PROFILING
    family_loglikes_ = measurement_model_->Evaluate(particles_, occlusion_indices_, update_occlusions);
    MEASURE("observation_model_->Evaluate")
}


void CoordinateParticleFilter::Resample(const int &new_state_count)
{
    size_t state_count;
    if(new_state_count == -1)
        state_count = particles_.size();
    else
        state_count = new_state_count;

    // todo we are not taking multiplicity into account
    std::vector<State > new_states(state_count);
    std::vector<double> new_state_times(state_count);
    std::vector<float> new_loglikes(state_count);
    std::vector<float> new_weights(state_count);
    std::vector<size_t> new_occlusion_indices(state_count);

    hf::DiscreteSampler sampler(family_loglikes_);

    for(size_t i = 0; i < state_count; i++)
    {
        size_t index = sampler.Sample();
        new_weights[i] = log_weights_[index];
        new_states[i] = particles_[index];
        new_state_times[i] = particle_times_[index];
        new_loglikes[i] = family_loglikes_[index]; // should we set this to 1?
        new_occlusion_indices[i] = occlusion_indices_[index];
    }

    particles_    		= new_states;
    particle_times_    = new_state_times;
    family_loglikes_= new_loglikes;
    occlusion_indices_ = new_occlusion_indices;
    log_weights_ = new_weights;




    // TODO: all of this is disgusting. i have to make sure that there is a consistent
    // representation of the hyperstate at all times!
    CoordinateParticleFilter::StateDistribution::Weights weights(particles_.size());
//    for(size_t i = 0; i < particles_.size(); i++)
//        weights(i) = parent_multiplicities_[i];
//    weights /= weights.sum();

    state_distribution_.SetDeltas(particles_, weights);
}



CoordinateParticleFilter::StateDistribution &CoordinateParticleFilter::stateDistribution()
{
    return state_distribution_;
}

//size_t CoordinateParticleFilter::control_size()
//{
//    return process_model_->control_size();
//}


// set and get fcts ==========================================================================================================================================================================================================================================================================================================================================================================================
void CoordinateParticleFilter::get(MeasurementModelPtr &observation_model) const
{
    observation_model = measurement_model_;
}
void CoordinateParticleFilter::get(ProcessModelPtr &process_model) const
{
    process_model = process_model_;
}
void CoordinateParticleFilter::get(std::vector<State >& states) const
{
    states = particles_;
}
void CoordinateParticleFilter::get(std::vector<double>& state_times) const
{
    state_times = particle_times_;
}
void CoordinateParticleFilter::get(std::vector<float>& loglikes) const
{
    loglikes = family_loglikes_;
}
const CoordinateParticleFilter::State& CoordinateParticleFilter::get_state(size_t index) const
{
    return particles_[index];
}
const std::vector<float> CoordinateParticleFilter::get_occlusions(size_t index) const
{
    return measurement_model_->get_occlusions(occlusion_indices_[index]	);
}
void CoordinateParticleFilter::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                             std::vector<std::vector<float> > &depth)
{
    measurement_model_->get_depth_values(intersect_indices, depth);
}

void CoordinateParticleFilter::set_independence(const std::vector<std::vector<size_t> >& independent_blocks)
{
    independent_blocks_ = independent_blocks;
}
void CoordinateParticleFilter::set(const MeasurementModelPtr &observation_model)
{
    measurement_model_ = observation_model;
}
void CoordinateParticleFilter::set(const ProcessModelPtr &process_model)
{
    process_model_ = process_model;
}


