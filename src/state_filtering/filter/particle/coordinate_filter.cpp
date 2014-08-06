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

RaoBlackwellCoordinateParticleFilter::RaoBlackwellCoordinateParticleFilter(const MeasurementModelPtr observation_model,
                                                   const ProcessModelPtr process_model,
                                                   const std::vector<std::vector<IndexType> > &sampling_blocks,
                                                   const ScalarType &max_kl_divergence):
    measurement_model_(observation_model),
    process_model_(process_model),
    max_kl_divergence_(max_kl_divergence)
{
    SamplingBlocks(sampling_blocks);
}
RaoBlackwellCoordinateParticleFilter::~RaoBlackwellCoordinateParticleFilter() { }


void RaoBlackwellCoordinateParticleFilter::Filter(const Measurement& measurement,
                                      const ScalarType&  delta_time,
                                      const InputType& input)
{
    INIT_PROFILING;
    measurement_model_->Measurement(measurement, delta_time);

    loglikes_ = std::vector<float>(samples_.size(), 0);
    noises_ = std::vector<NoiseType>(samples_.size(), NoiseType::Zero(dimension_));
    next_samples_ = samples_;

    for(size_t block_index = 0; block_index < sampling_blocks_.size(); block_index++)
    {
        for(size_t particle_index = 0; particle_index < samples_.size(); particle_index++)
        {
            for(size_t i = 0; i < sampling_blocks_[block_index].size(); i++)
                noises_[particle_index](sampling_blocks_[block_index][i]) = unit_gaussian_.Sample()(0);

            process_model_->Condition(delta_time,
                                      samples_[particle_index],
                                      input);
            next_samples_[particle_index] = process_model_->MapGaussian(noises_[particle_index]);
        }

        bool update_occlusions = (block_index == sampling_blocks_.size()-1);
        cout << "evaluating with " << next_samples_.size() << " samples " << endl;
        RESET;
        std::vector<float> new_loglikes = measurement_model_->Loglikes(next_samples_,
                                                                       indices_,
                                                                       update_occlusions);
        MEASURE("evaluation");
        std::vector<float> delta_loglikes(new_loglikes.size());
        for(size_t i = 0; i < delta_loglikes.size(); i++)
            delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
        loglikes_ = new_loglikes;
        UpdateWeights(delta_loglikes);
    }

    samples_ = next_samples_;
    state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
}

void RaoBlackwellCoordinateParticleFilter::Resample(const IndexType& sample_count)
{
    std::vector<StateType> samples(sample_count);
    std::vector<IndexType> indices(sample_count);
    std::vector<NoiseType> noises(sample_count);
    std::vector<StateType> next_samples(sample_count);
    std::vector<float> loglikes(sample_count);

    hf::DiscreteSampler sampler(log_weights_);

    for(IndexType i = 0; i < sample_count; i++)
    {
        IndexType index = sampler.Sample();

        samples[i]      = samples_[index];
        indices[i]      = indices_[index];
        noises[i]       = noises_[index];
        next_samples[i] = next_samples_[index];
        loglikes[i]     = loglikes_[index];
    }
    samples_        = samples;
    indices_        = indices;
    noises_         = noises;
    next_samples_   = next_samples;
    loglikes_       = loglikes;

    log_weights_        = std::vector<float>(samples_.size(), 0.);

    state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
}

void RaoBlackwellCoordinateParticleFilter::UpdateWeights(std::vector<float> log_weight_diffs)
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

    cout << "kl divergence: " << kl_divergence << " max divergence: " << max_kl_divergence_ << endl;
    if(kl_divergence > max_kl_divergence_)
        Resample(samples_.size());
}

// set
void RaoBlackwellCoordinateParticleFilter::Samples(const std::vector<StateType >& samples)
{
    // we copy the new states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    samples_ = samples;
    indices_ = vector<size_t>(samples_.size(), 0); measurement_model_->Reset();
    log_weights_ = std::vector<float>(samples_.size(), 0);
}
void RaoBlackwellCoordinateParticleFilter::SamplingBlocks(const std::vector<std::vector<IndexType> >& sampling_blocks)
{
    sampling_blocks_ = sampling_blocks;

    // make sure sizes are consistent
    dimension_ = 0;
    for(size_t i = 0; i < sampling_blocks_.size(); i++)
        for(size_t j = 0; j < sampling_blocks_[i].size(); j++)
            dimension_++;

    // TODO: COMPARE THIS TO NOISE DIMENSION OF THE GAUSSIAN MAPPABLE PROCESS
}

// get
const std::vector<RaoBlackwellCoordinateParticleFilter::StateType>& RaoBlackwellCoordinateParticleFilter::Samples() const
{
    return samples_;
}
RaoBlackwellCoordinateParticleFilter::StateDistributionType &RaoBlackwellCoordinateParticleFilter::StateDistribution()
{
    return state_distribution_;
}
