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

//#include "image_visualizer.hpp"
#include <omp.h>
#include <string>

#include <boost/lexical_cast.hpp>


/// this namespace contains all the filters
namespace distributions
{

template<typename ScalarType_, typename StateType_, typename MeasurementType_, int NOISE_DIMENSION_EIGEN>
class RaoBlackwellCoordinateParticleFilter
{
    // TODO: CHECK THAT PROCESS DERIVES FROM STATIONARY AND GAUSSIAN MAPPABLE
    // AND MEASUREMENT MODEL FROM RAOBLACKWELLMEASUREMENT MODEL
public:
//    typedef typename ProcessType::ScalarType ScalarType;
    typedef double ScalarType;
    typedef size_t IndexType;
    typedef FloatingBodySystem<-1> StateType;

    // TODO: MAKE SURE THAT STATE TYPE FROM OBSERVATION MODEL AND PROCESS MODEL ARE THE SAME



    typedef Eigen::VectorXd InputType;


    typedef Eigen::Matrix<ScalarType, -1, -1>   MeasurementType;


    typedef distributions::BrownianObjectMotion<ScalarType, -1> ProcessModel;

    typedef distributions::RaoBlackwellMeasurementModel<ScalarType, StateType, MeasurementType> MeasurementModel;


    typedef SumOfDeltas<ScalarType, StateType> StateDistributionType;

    typedef MeasurementModel::MeasurementType Measurement;



    typedef Eigen::VectorXd NoiseType;




    typedef boost::shared_ptr<RaoBlackwellCoordinateParticleFilter> Ptr;

    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;
    typedef boost::shared_ptr<MeasurementModel> MeasurementModelPtr;

public:
    RaoBlackwellCoordinateParticleFilter(const MeasurementModelPtr observation_model,
                            const ProcessModelPtr process_model,
                            const std::vector<std::vector<IndexType> >& sampling_blocks,
                            const ScalarType& max_kl_divergence = 0):
        measurement_model_(observation_model),
        process_model_(process_model),
        max_kl_divergence_(max_kl_divergence)
    {
        SamplingBlocks(sampling_blocks);
    }
    virtual ~RaoBlackwellCoordinateParticleFilter() {}

public:
    // main functions
    void Filter(const Measurement& measurement,
                const ScalarType&  delta_time,
                const InputType&   input)
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
            std::cout << "evaluating with " << next_samples_.size() << " samples " << std::endl;
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

    void Resample(const IndexType& sample_count)
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

private:
    void UpdateWeights(std::vector<float> log_weight_diffs)
    {
        for(size_t i = 0; i < log_weight_diffs.size(); i++)
            log_weights_[i] += log_weight_diffs[i];

        std::vector<float> weights = log_weights_;
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
        log_weights_ = std::vector<float>(samples_.size(), 0);
    }

    void SamplingBlocks(const std::vector<std::vector<IndexType> >& sampling_blocks)
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
    const std::vector<StateType>& Samples() const
    {
        return samples_;
    }
    StateDistributionType& StateDistribution()
    {
        return state_distribution_;
    }

private:
    unsigned dimension_;

    // TODO this is not used properly yet
    StateDistributionType state_distribution_;


    // TODO: THE FLOAT SHOULD ALSO BE SCALAR TYPE?
    std::vector<StateType > samples_;
    std::vector<IndexType> indices_;
    std::vector<float>  log_weights_;

    // TODO: THESE DO NOT HAVE TO BE MEMBERS
    std::vector<NoiseType> noises_;
    std::vector<StateType> next_samples_;
    std::vector<float> loglikes_;


    // process and observation models
    MeasurementModelPtr measurement_model_;
    ProcessModelPtr process_model_;

    // distribution for sampling
    Gaussian<ScalarType, 1> unit_gaussian_;

    // parameters
    std::vector<std::vector<IndexType> > sampling_blocks_;
    ScalarType max_kl_divergence_;
};

}

#endif
