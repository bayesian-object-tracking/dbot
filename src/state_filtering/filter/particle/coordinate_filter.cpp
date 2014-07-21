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
                                   const std::vector<std::vector<size_t> >& independent_blocks,
                                   const double &max_kl_divergence):
    measurement_model_(observation_model),
    process_model_(process_model),
    independent_blocks_(independent_blocks),
    state_distribution_(process_model->variable_size()),
    max_kl_divergence_(max_kl_divergence)
{
    // make sure sizes are consistent
    size_t sample_size = 0;
    for(size_t i = 0; i < independent_blocks_.size(); i++)
        for(size_t j = 0; j < independent_blocks_[i].size(); j++)
            sample_size++;

    if(sample_size != process_model_->sample_size())
    {
        cout << "the number of dof in the dependency specification does not correspond to" <<
                " to the number of dof in the process model!!" << endl;
        exit(-1);
    }

    if(measurement_model_->state_size() != process_model_->variable_size())
    {
        cout << "the process and the measurement model do not have the same state size!!" << endl;
        exit(-1);
    }



}
CoordinateFilter::~CoordinateFilter() { }

// create offspring
void CoordinateFilter::PartialPropagate(const Control& control,
                                        const double& observation_time)
{
    INIT_PROFILING;
    // we write to the following members ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.resize(particles_.size());
    partial_children_.resize(particles_.size());
    partial_children_occlusion_indices_.resize(particles_.size());
    zero_children_.resize(particles_.size());

    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
    {
        // propagation with zero noise is required for normalization ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        process_model_->conditional(observation_time - particle_times_[state_index], particles_[state_index], control);
        zero_children_[state_index] = process_model_->mapNormal(Noise::Zero(process_model_->sample_size()));

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
                Noise partial_noise(Noise::Zero(process_model_->sample_size()));
                for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                    partial_noise(independent_blocks_[block_index][i]) = unit_gaussian_.sample()(0);

                partial_noises_[state_index][block_index][mult_index] = partial_noise;
                partial_children_[state_index][block_index][mult_index] = process_model_->mapNormal(partial_noise);
                partial_children_occlusion_indices_[state_index][block_index][mult_index] = occlusion_indices_[state_index];
            }
        }
    }
    MEASURE("creating samples")
}







// evaluate offspring
void CoordinateFilter::PartialEvaluate(const Measurement& observation,
                                       const double& observation_time)
{
    // compute partial_loglikes_ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    measurement_model_->measurement(observation, observation_time);
    hf::Structurer3D converter;
    vector<size_t> flat_partial_children_occlusion_indices_ = converter.Flatten(partial_children_occlusion_indices_);

    INIT_PROFILING;
    partial_children_loglikes_ = converter.Deepen(
                measurement_model_->Evaluate(converter.Flatten(partial_children_),
                                             flat_partial_children_occlusion_indices_, false));
    MEASURE("evaluation with " + lexical_cast<string>(flat_partial_children_occlusion_indices_.size()) + " samples");


    vector<float> zero_loglikes = measurement_model_->Evaluate(zero_children_, occlusion_indices_, false);
    MEASURE("evaluation with " + lexical_cast<string>(zero_children_.size()) + " samples");

    // compute family_loglikes_ ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    family_loglikes_.resize(particles_.size());
    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
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

void CoordinateFilter::PartialResample(const Control& control,
                                       const double& observation_time,
                                       const size_t &new_n_states)
{
    // find the children counts per parent ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    INIT_PROFILING;
    hf::DiscreteSampler sampler(family_loglikes_);
    vector<size_t> children_counts(particles_.size(), 0);
    for(size_t new_state_index = 0; new_state_index < new_n_states; new_state_index++)
        children_counts[sampler.Sample()]++;

    MEASURE("sampling children_counts");

    // combine partial children into children ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<State> children;
    std::vector<double> children_times;
    vector<size_t> children_multiplicities;
    vector<size_t> children_occlusion_indices;
    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
    {
        if(children_counts[state_index] == 0) continue;
        vector<vector<size_t> >	mult_indices(children_counts[state_index], vector<size_t>(independent_blocks_.size()));
        for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
        {
            hf::DiscreteSampler sampler(partial_children_loglikes_[state_index][block_index]);

            int sample = sampler.Sample();
            for(size_t child_index = 0; child_index < children_counts[state_index]; child_index++)
                mult_indices[child_index][block_index] = sampler.Sample();
        }
        vector<size_t> child_multiplicities;
        hf::SortAndCollapse(mult_indices, child_multiplicities);

        process_model_->conditional(observation_time - particle_times_[state_index], particles_[state_index], control);
        for(size_t child_index = 0; child_index < mult_indices.size(); child_index++)
        {
            VectorXd noise(VectorXd::Zero(process_model_->sample_size()));
            for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
                noise += partial_noises_[state_index][block_index][mult_indices[child_index][block_index]];

            children.push_back(process_model_->mapNormal(noise));
            children_times.push_back(observation_time);
            children_multiplicities.push_back(child_multiplicities[child_index]);
            children_occlusion_indices.push_back(occlusion_indices_[state_index]);
        }
    }
    MEASURE("sampling children");

    // children become new parents ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    particles_ = children;
    particle_times_ = children_times;
    parent_multiplicities_ = children_multiplicities;
    occlusion_indices_ = children_occlusion_indices;
    family_loglikes_ = vector<float>(particles_.size(), 0);

    // new parents have no children yet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.clear();
    partial_children_.clear();
    partial_children_occlusion_indices_.clear();
    zero_children_.clear();
    partial_children_loglikes_.clear();
}


void CoordinateFilter::UpdateOcclusions(const Measurement& observation,
                                        const double& observation_time)
{
    measurement_model_->measurement(observation, observation_time);
    INIT_PROFILING;
    measurement_model_->Evaluate(particles_, occlusion_indices_, true);
    MEASURE("evaluation with " + lexical_cast<string>(particles_.size()) + " samples")

}






void CoordinateFilter::Enchilada(
        const Control control,
        const double &observation_time,
        const Measurement& observation,
        const size_t &new_n_states)
{
    vector<size_t> sorted_multiplicities = parent_multiplicities_;
    hf::SortDescend(sorted_multiplicities, sorted_multiplicities);
    cout << "multiplicites: ";
    hf::PrintVector(sorted_multiplicities);

    PartialPropagate(control, observation_time);
    PartialEvaluate(observation, observation_time);
    PartialResample(control, observation_time, new_n_states);
    UpdateOcclusions(observation, observation_time);

    state_distribution_.setDeltas(particles_); // not sure whether this is the right place
}






void CoordinateFilter::Enchiladisima(const Control control,
        const double &observation_time,
        const Measurement& observation, const size_t &evaluation_count, const size_t &factor_evaluation_count)
{
    size_t M = factor_evaluation_count;
    size_t D = independent_blocks_.size();
    size_t N = evaluation_count / (M*D);

    cout << "number of samples per factor M = " << M  << endl
         << "number of factors D = " << D << endl
         << "number of particles N = " << N << endl
         << "total number of evaluations M*D*N = " << N*D*M << endl;


    if(particles_.size() != N || family_loglikes_.size() != N ||
            particle_times_.size() != N || occlusion_indices_.size() != N||
            log_weights_.size() != N)
    {
        cout << "parents_.size() = " << particles_.size() << endl;
        cout << "family_loglikes_.size () = " << family_loglikes_.size() << endl;
        cout << "parent_times_.size () = " << particle_times_.size() << endl;
        cout << "parent_occlusion_indices_.size () = " << occlusion_indices_.size() << endl;
        cout << "log_weights_.size () = " << log_weights_.size() << endl;



        cout << "N " << N << endl;
        cout << "D " << D << endl;
        exit(-1);
    }


    INIT_PROFILING;
    // we write to the following members ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.resize(N);
    partial_children_.resize(N);
    partial_children_occlusion_indices_.resize(N);


    for(size_t state_index = 0; state_index < N; state_index++)
    {
        // propagation with zero noise is required for normalization ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        process_model_->conditional(observation_time - particle_times_[state_index], particles_[state_index], control);

        // fill the partial noises and resulting states --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        partial_noises_[state_index].resize(D);
        partial_children_[state_index].resize(D);
        partial_children_occlusion_indices_[state_index].resize(D);
        for(size_t block_index = 0; block_index < D; block_index++)
        {
            partial_noises_[state_index][block_index].resize(M);
            partial_children_[state_index][block_index].resize(M);
            partial_children_occlusion_indices_[state_index][block_index].resize(M);
            for(size_t m = 0; m < M; m++)
            {
                Noise partial_noise(Noise::Zero(process_model_->sample_size()));
                for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                    partial_noise(independent_blocks_[block_index][i]) = unit_gaussian_.sample()(0);

                partial_noises_[state_index][block_index][m] = partial_noise;
                partial_children_[state_index][block_index][m] = process_model_->mapNormal(partial_noise);
                partial_children_occlusion_indices_[state_index][block_index][m] = occlusion_indices_[state_index];
            }
        }
    }
    MEASURE("creating samples");






    // compute partial_loglikes_ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    measurement_model_->measurement(observation, observation_time);
    hf::Structurer3D converter;
    vector<size_t> flat_partial_children_occlusion_indices_ = converter.Flatten(partial_children_occlusion_indices_);

    RESET;
    partial_children_loglikes_ = converter.Deepen(
                measurement_model_->Evaluate(converter.Flatten(partial_children_),
                                             flat_partial_children_occlusion_indices_, false));
    MEASURE("evaluation with " + lexical_cast<string>(flat_partial_children_occlusion_indices_.size()) + " samples");





    std::vector<float> proposal_loglikes(N);

    // propagate the shizzle
    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
    {
        Noise noise(Noise::Zero(process_model_->sample_size()));
        proposal_loglikes[state_index] = -float(D-1) * family_loglikes_[state_index];
        for(size_t block_index = 0; block_index < D; block_index++)
        {
            hf::DiscreteSampler sampler(partial_children_loglikes_[state_index][block_index]);
            size_t sample = sampler.Sample();

            noise += partial_noises_[state_index][block_index][sample];
            proposal_loglikes[state_index] += partial_children_loglikes_[state_index][block_index][sample];
        }

        process_model_->conditional(observation_time - particle_times_[state_index], particles_[state_index], control);
        particles_[state_index] = process_model_->mapNormal(noise);
        particle_times_[state_index] = observation_time;
    }

    // new parents have no children yet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.clear();
    partial_children_.clear();
    partial_children_occlusion_indices_.clear();
    partial_children_loglikes_.clear();



    RESET;
    family_loglikes_ = measurement_model_->Evaluate(particles_, occlusion_indices_, true);
    MEASURE("evaluation with " + lexical_cast<string>(particles_.size()) + " samples");

    for(size_t i = 0; i < N; i++)
        log_weights_[i] += family_loglikes_[i] - proposal_loglikes[i];


    vector<float> sorted_weights = log_weights_;
    hf::Sort(sorted_weights, 1);

    for(int i = sorted_weights.size() - 1; i >= 0; i--)
        sorted_weights[i] -= sorted_weights[0];

    sorted_weights = hf::Apply<float, float>(sorted_weights, std::exp);

    sorted_weights = hf::SetSum(sorted_weights, float(1));

    cout << "weights " << endl;
    hf::PrintVector(sorted_weights);



    std::vector<State > new_states(N);
    std::vector<double> new_state_times(N);
    std::vector<float> new_loglikes(N);
    std::vector<size_t> new_occlusion_indices(N);

    hf::DiscreteSampler sampler(log_weights_); //TESTING

    for(size_t i = 0; i < N; i++)
    {
        size_t index = sampler.Sample();
        new_states[i] = particles_[index];
        new_state_times[i] = particle_times_[index];
        new_loglikes[i] = family_loglikes_[index]; // should we set this to 1?
        new_occlusion_indices[i] = occlusion_indices_[index];
    }

    log_weights_ = std::vector<float>(N, 0);
    particles_    		= new_states;
    particle_times_    = new_state_times;
    family_loglikes_ = new_loglikes;
    occlusion_indices_ = new_occlusion_indices;


























    state_distribution_.setDeltas(particles_); // not sure whether this is the right place
}





























void CoordinateFilter::UpdateWeights(std::vector<float> log_weight_diffs)
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











void CoordinateFilter::Enchiladisimimisima( const Control control,
                                            const double &observation_time,
                                            const Measurement& observation)
{
    INIT_PROFILING;
    measurement_model_->measurement(observation, observation_time);

    loglikes_ = std::vector<float>(particles_.size(), 0);
    noises_ = std::vector<Noise>(particles_.size(), Noise::Zero(process_model_->sample_size()));
    propagated_particles_ = std::vector<State>(particles_.size());

    for(size_t block_index = 0; block_index < independent_blocks_.size(); block_index++)
    {
        for(size_t particle_index = 0; particle_index < particles_.size(); particle_index++)
        {
            for(size_t i = 0; i < independent_blocks_[block_index].size(); i++)
                noises_[particle_index](independent_blocks_[block_index][i]) = unit_gaussian_.sample()(0);

            process_model_->conditional(observation_time - particle_times_[particle_index],
                                        particles_[particle_index],
                                        control);
            propagated_particles_[particle_index] = process_model_->mapNormal(noises_[particle_index]);
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

    state_distribution_.setDeltas(particles_); // not sure whether this is the right place
}








void CoordinateFilter::set_states(const std::vector<State >& states,
                                  const std::vector<double>& state_times,
                                  const std::vector<size_t>& multiplicities,
                                  const std::vector<float> &loglikes)
{
    // we copy the new states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    particles_ = states;
    particle_times_ = state_times;
    parent_multiplicities_ = multiplicities;
    occlusion_indices_ = vector<size_t>(particles_.size(), 0); measurement_model_->set_occlusions();
    family_loglikes_ = loglikes;

    // if some arguments have not been passed we set to default values ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(particle_times_.empty()) particle_times_ = std::vector<double>(particles_.size(), 0);
    if(parent_multiplicities_.empty()) parent_multiplicities_ = std::vector<size_t>(particles_.size(), 1);
    if(family_loglikes_.empty()) family_loglikes_ = std::vector<float>(particles_.size(), 0);
    log_weights_ = std::vector<float>(particles_.size(), 0);


    // new parents have no children yet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    partial_noises_.clear();
    partial_children_.clear();
    partial_children_occlusion_indices_.clear();
    zero_children_.clear();
    partial_children_loglikes_.clear();


}





































void CoordinateFilter::Propagate(
        const Control control,
        const double &current_time)
{
    // we propagate the states to the current time, appying the control input ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for(size_t state_index = 0; state_index < particles_.size(); state_index++)
    {        
        process_model_->conditional(current_time - particle_times_[state_index], particles_[state_index], control);
        particles_[state_index] = process_model_->sample();
        particle_times_[state_index] = current_time;
    }
}


void CoordinateFilter::Evaluate(
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


void CoordinateFilter::Resample(const int &new_state_count)
{
    size_t state_count;
    if(new_state_count == -1)
        state_count = particles_.size();
    else
        state_count = new_state_count;

    // todo we are not taking multiplicity into account
    std::vector<State > new_states(state_count);
    std::vector<double> new_state_times(state_count);
    std::vector<size_t> new_multiplicities(state_count);
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
        new_multiplicities[i] = parent_multiplicities_[index];
        new_loglikes[i] = family_loglikes_[index]; // should we set this to 1?
        new_occlusion_indices[i] = occlusion_indices_[index];
    }

    particles_    		= new_states;
    particle_times_    = new_state_times;
    parent_multiplicities_ = new_multiplicities;
    family_loglikes_= new_loglikes;
    occlusion_indices_ = new_occlusion_indices;
    log_weights_ = new_weights;




    // TODO: all of this is disgusting. i have to make sure that there is a consistent
    // representation of the hyperstate at all times!
    CoordinateFilter::StateDistribution::Weights weights(particles_.size());
    for(size_t i = 0; i < particles_.size(); i++)
        weights(i) = parent_multiplicities_[i];
    weights /= weights.sum();

    state_distribution_.setDeltas(particles_, weights);
}


void CoordinateFilter::Sort()
{
    vector<int> indices = hf::SortDescend(family_loglikes_);

    std::vector<State > sorted_states(indices.size());
    std::vector<double> sorted_state_times(indices.size());
    std::vector<size_t> sorted_multiplicities(indices.size());
    std::vector<float> sorted_loglikes(indices.size());
    std::vector<size_t> sorted_occlusion_indices(indices.size());
    for(size_t i = 0; i < indices.size(); i++)
    {
        sorted_states[i] = particles_[indices[i]];
        sorted_state_times[i] = particle_times_[indices[i]];
        sorted_multiplicities[i] = parent_multiplicities_[indices[i]];
        sorted_loglikes[i] = family_loglikes_[indices[i]]; // should we set this to 1?
        sorted_occlusion_indices[i] = occlusion_indices_[indices[i]];
    }

    particles_    		= sorted_states;
    particle_times_    = sorted_state_times;
    parent_multiplicities_ = sorted_multiplicities;
    family_loglikes_= sorted_loglikes;
    occlusion_indices_ = sorted_occlusion_indices;
}




CoordinateFilter::StateDistribution &CoordinateFilter::stateDistribution()
{
    return state_distribution_;
}

size_t CoordinateFilter::control_size()
{
    return process_model_->control_size();
}


// set and get fcts ==========================================================================================================================================================================================================================================================================================================================================================================================
void CoordinateFilter::get(MeasurementModelPtr &observation_model) const
{
    observation_model = measurement_model_;
}
void CoordinateFilter::get(ProcessModelPtr &process_model) const
{
    process_model = process_model_;
}
void CoordinateFilter::get(std::vector<State >& states) const
{
    states = particles_;
}
void CoordinateFilter::get(std::vector<double>& state_times) const
{
    state_times = particle_times_;
}
void CoordinateFilter::get(std::vector<size_t>& multiplicities) const
{
    multiplicities = parent_multiplicities_;
}
void CoordinateFilter::get(std::vector<float>& loglikes) const
{
    loglikes = family_loglikes_;
}
const CoordinateFilter::State& CoordinateFilter::get_state(size_t index) const
{
    return particles_[index];
}
const std::vector<float> CoordinateFilter::get_occlusions(size_t index) const
{
    return measurement_model_->get_occlusions(occlusion_indices_[index]	);
}
void CoordinateFilter::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                             std::vector<std::vector<float> > &depth)
{
    measurement_model_->get_depth_values(intersect_indices, depth);
}

void CoordinateFilter::set_independence(const std::vector<std::vector<size_t> >& independent_blocks)
{
    independent_blocks_ = independent_blocks;
}
void CoordinateFilter::set(const MeasurementModelPtr &observation_model)
{
    measurement_model_ = observation_model;
}
void CoordinateFilter::set(const ProcessModelPtr &process_model)
{
    process_model_ = process_model;
}


