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

#ifndef COORDINATE_FILTER_
#define COORDINATE_FILTER_

#include <vector>
#include <limits>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <state_filtering/models/measurement/features/rao_blackwell_measurement_model.hpp>
//#include <state_filtering/models/measurement/cpu_image_observation_modegaussian_pixel_observation_modelel.hpp>
//#include <state_filtering/models/process/implementations/occlusion_process.hpp>

#include <state_filtering/utils/rigid_body_renderer.hpp>
#include <state_filtering/models/process/features/stationary_process.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>
#include <state_filtering/distributions/implementations/sum_of_deltas.hpp>

#include <state_filtering/states/floating_body_system.hpp>

#include <state_filtering/models/process/implementations/brownian_object_motion.hpp>

#include <state_filtering/filters/stochastic/coordinate_filter.hpp>

#include <state_filtering/utils/macros.hpp>
#include <state_filtering/utils/helper_functions.hpp>

#include <state_filtering/distributions/features/gaussian_mappable.hpp>

//#include "image_visualizer.hpp"
#include <omp.h>
#include <string>

#include <boost/lexical_cast.hpp>


/// this namespace contains all the filters
namespace distributions
{

template<typename ScalarType_, typename StateType_, typename InputType_,
         typename MeasurementType_, typename MeasurementStateType_,typename IndexType_, int NOISE_DIMENSION_EIGEN>
class RaoBlackwellCoordinateParticleFilter
{
public:
    // basic types
    typedef ScalarType_         ScalarType;
    typedef StateType_          StateType;
    typedef InputType_          InputType;
    typedef MeasurementType_    MeasurementType;
    typedef IndexType_          IndexType;

    // process model
    typedef StationaryProcess<ScalarType, StateType, InputType>             StationaryType;
    typedef GaussianMappable<ScalarType, StateType, NOISE_DIMENSION_EIGEN>  MappableType;
    typedef typename MappableType::NoiseType                                NoiseType;

    // measurement model
    typedef distributions::RaoBlackwellMeasurementModel
            <ScalarType, MeasurementStateType_, MeasurementType, IndexType>     MeasurementModelType;

    // state distribution
    typedef SumOfDeltas<ScalarType, StateType>                      StateDistributionType;

public:
    template<typename ProcessPointer, typename MeasurementModelPointer>
    RaoBlackwellCoordinateParticleFilter(const ProcessPointer           process_model,
                                         const MeasurementModelPointer  observation_model,
                                         const std::vector<std::vector<IndexType> >& sampling_blocks,
                                         const ScalarType& max_kl_divergence = 0):
        measurement_model_(observation_model),
        stationary_process_(process_model),
        mappable_process_(process_model),
        max_kl_divergence_(max_kl_divergence)
    {
        SamplingBlocks(sampling_blocks);
    }
    virtual ~RaoBlackwellCoordinateParticleFilter() {}

public:
    void Filter(const MeasurementType& measurement,
                const ScalarType&  delta_time,
                const InputType&   input)
    {
        INIT_PROFILING;
        measurement_model_->Measurement(measurement, delta_time);

        loglikes_ = std::vector<ScalarType>(samples_.size(), 0);
        noises_ = std::vector<NoiseType>(samples_.size(), NoiseType::Zero(mappable_process_->NoiseDimension()));
        next_samples_ = samples_;

        for(size_t block_index = 0; block_index < sampling_blocks_.size(); block_index++)
        {
            for(size_t particle_index = 0; particle_index < samples_.size(); particle_index++)
            {
                for(size_t i = 0; i < sampling_blocks_[block_index].size(); i++)
                    noises_[particle_index](sampling_blocks_[block_index][i]) = unit_gaussian_.Sample()(0);

                stationary_process_->Condition(delta_time,
                                          samples_[particle_index],
                                          input);
                next_samples_[particle_index] = mappable_process_->MapGaussian(noises_[particle_index]);
            }

            bool update_occlusions = (block_index == sampling_blocks_.size()-1);
            std::cout << "evaluating with " << next_samples_.size() << " samples " << std::endl;
            RESET;
            std::vector<ScalarType> new_loglikes = measurement_model_->Loglikes(next_samples_,
                                                                           indices_,
                                                                           update_occlusions);
            MEASURE("evaluation");
            std::vector<ScalarType> delta_loglikes(new_loglikes.size());
            for(size_t i = 0; i < delta_loglikes.size(); i++)
                delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
            loglikes_ = new_loglikes;
            UpdateWeights(delta_loglikes);
        }

        samples_ = next_samples_;
        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

    void Resample(const IndexType& sample_count)
    {
        std::vector<StateType> samples(sample_count);
        std::vector<IndexType> indices(sample_count);
        std::vector<NoiseType> noises(sample_count);
        std::vector<StateType> next_samples(sample_count);
        std::vector<ScalarType> loglikes(sample_count);

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

        log_weights_        = std::vector<ScalarType>(samples_.size(), 0.);

        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

private:
    void UpdateWeights(std::vector<ScalarType> log_weight_diffs)
    {
        for(size_t i = 0; i < log_weight_diffs.size(); i++)
            log_weights_[i] += log_weight_diffs[i];

        std::vector<ScalarType> weights = log_weights_;
        hf::Sort(weights, 1);

        for(int i = weights.size() - 1; i >= 0; i--)
            weights[i] -= weights[0];

        weights = hf::Apply<ScalarType, ScalarType>(weights, std::exp);
        weights = hf::SetSum(weights, ScalarType(1));

        // compute KL divergence to uniform distribution KL(p|u)
        ScalarType kl_divergence = std::log(ScalarType(weights.size()));
        for(size_t i = 0; i < weights.size(); i++)
        {
            ScalarType information = - std::log(weights[i]) * weights[i];
            if(!std::isfinite(information))
                information = 0; // the limit for weight -> 0 is equal to 0
            kl_divergence -= information;
        }

        std::cout << "kl divergence: " << kl_divergence << " max divergence: " << max_kl_divergence_ << std::endl;
        if(kl_divergence > max_kl_divergence_)
            Resample(samples_.size());
    }

public:
    // set
    void Samples(const std::vector<StateType >& samples)
    {
        samples_ = samples;
        indices_ = std::vector<size_t>(samples_.size(), 0); measurement_model_->Reset();
        log_weights_ = std::vector<ScalarType>(samples_.size(), 0);
    }
    void SamplingBlocks(const std::vector<std::vector<IndexType> >& sampling_blocks)
    {
        sampling_blocks_ = sampling_blocks;

        // make sure sizes are consistent
        IndexType dimension = 0;
        for(size_t i = 0; i < sampling_blocks_.size(); i++)
            for(size_t j = 0; j < sampling_blocks_[i].size(); j++)
                dimension++;

        if(dimension != mappable_process_->NoiseDimension())
        {
            std::cout << "the dimension of the sampling blocks is " << dimension
                      << " while the dimension of the noise is "
                      << mappable_process_->NoiseDimension() << std::endl;
            exit(-1);
        }
    }

    // get
    const std::vector<StateType>& Samples() const
    {
        return samples_;
    }
    StateDistributionType& StateDistribution()
    {
        return state_distribution_;
    }

private:
    // internal state TODO: THIS COULD BE MADE MORE COMPACT!!
    StateDistributionType state_distribution_;

    std::vector<StateType > samples_;
    std::vector<IndexType> indices_;
    std::vector<ScalarType>  log_weights_;
    std::vector<NoiseType> noises_;
    std::vector<StateType> next_samples_;
    std::vector<ScalarType> loglikes_;


    // measurement model
    boost::shared_ptr<MeasurementModelType> measurement_model_;

    // process model
    boost::shared_ptr<StationaryType>   stationary_process_;
    boost::shared_ptr<MappableType>     mappable_process_;

    // parameters
    std::vector<std::vector<IndexType> > sampling_blocks_;
    ScalarType max_kl_divergence_;

    // distribution for sampling
    Gaussian<ScalarType, 1> unit_gaussian_;
};

}

#endif
