/*************************************************************************
This software allows for filtering in high-dimensional observation and
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

#ifndef STATE_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP
#define STATE_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP

#include <vector>
#include <limits>
#include <string>

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <state_filtering/utils/macros.hpp>
#include <state_filtering/utils/traits.hpp>
#include <state_filtering/utils/helper_functions.hpp>
#include <state_filtering/distributions/gaussian.hpp>
#include <state_filtering/distributions/sum_of_deltas.hpp>
#include <state_filtering/distributions/interfaces/gaussian_mappable_interface.hpp>
#include <state_filtering/models/observers/interfaces/rao_blackwell_observer.hpp>
#include <state_filtering/models/processes/interfaces/stationary_process_interface.hpp>

namespace sf
{

template<typename ProcessModel, typename ObservationModel>
class RaoBlackwellCoordinateParticleFilter
{
public:
    typedef typename internal::Traits<ProcessModel>::Scalar   Scalar;
    typedef typename internal::Traits<ProcessModel>::State    State;
    typedef typename internal::Traits<ProcessModel>::Input    Input;
    typedef typename internal::Traits<ProcessModel>::Noise    Noise;

    typedef typename ObservationModel::Observation            Observation;

    // state distribution
    typedef SumOfDeltas<State> StateDistributionType;

public:
    RaoBlackwellCoordinateParticleFilter(
            const boost::shared_ptr<ProcessModel> process_model,
            const boost::shared_ptr<ObservationModel>  observation_model,
            const std::vector<std::vector<size_t> >& sampling_blocks,
            const Scalar& max_kl_divergence = 0):
        observation_model_(observation_model),
        process_model_(process_model),
        max_kl_divergence_(max_kl_divergence)
    {
        SF_REQUIRE_INTERFACE(
            ProcessModel,
            StationaryProcessInterface<State, Input>);

        SF_REQUIRE_INTERFACE(
            ProcessModel,
            GaussianMappableInterface<State, Noise>);

        SF_REQUIRE_INTERFACE(
            ObservationModel,
            RaoBlackwellObserver<State, Observation>);

        SamplingBlocks(sampling_blocks);
    }

    virtual ~RaoBlackwellCoordinateParticleFilter() {}

public:
    void Filter(const Observation& observation,
                const Scalar&  delta_time,
                const Input&   input)
    {
        INIT_PROFILING;
        observation_model_->SetObservation(observation, delta_time);

        loglikes_ = std::vector<Scalar>(samples_.size(), 0);
        noises_ = std::vector<Noise>(samples_.size(), Noise::Zero(process_model_->NoiseDimension()));
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
            std::vector<Scalar> new_loglikes = observation_model_->Loglikes(next_samples_,
                                                                           indices_,
                                                                           update_occlusions);
            MEASURE("evaluation");
            std::vector<Scalar> delta_loglikes(new_loglikes.size());
            for(size_t i = 0; i < delta_loglikes.size(); i++)
                delta_loglikes[i] = new_loglikes[i] - loglikes_[i];
            loglikes_ = new_loglikes;
            UpdateWeights(delta_loglikes);
        }

        samples_ = next_samples_;
        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

    void Resample(const size_t& sample_count)
    {
        std::vector<State> samples(sample_count);
        std::vector<size_t> indices(sample_count);
        std::vector<Noise> noises(sample_count);
        std::vector<State> next_samples(sample_count);
        std::vector<Scalar> loglikes(sample_count);

        sf::hf::DiscreteSampler sampler(log_weights_);

        for(size_t i = 0; i < sample_count; i++)
        {
            size_t index = sampler.Sample();

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

        log_weights_        = std::vector<Scalar>(samples_.size(), 0.);

        state_distribution_.SetDeltas(samples_); // not sure whether this is the right place
    }

private:
    void UpdateWeights(std::vector<Scalar> log_weight_diffs)
    {
        for(size_t i = 0; i < log_weight_diffs.size(); i++)
            log_weights_[i] += log_weight_diffs[i];

        std::vector<Scalar> weights = log_weights_;
        sf::hf::Sort(weights, 1);

        for(int i = weights.size() - 1; i >= 0; i--)
            weights[i] -= weights[0];

        weights = sf::hf::Apply<Scalar, Scalar>(weights, std::exp);
        weights = sf::hf::SetSum(weights, Scalar(1));

        // compute KL divergence to uniform distribution KL(p|u)
        Scalar kl_divergence = std::log(Scalar(weights.size()));
        for(size_t i = 0; i < weights.size(); i++)
        {
            Scalar information = - std::log(weights[i]) * weights[i];
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
    void Samples(const std::vector<State >& samples)
    {
        samples_ = samples;
        indices_ = std::vector<size_t>(samples_.size(), 0); observation_model_->Reset();
        log_weights_ = std::vector<Scalar>(samples_.size(), 0);
    }
    void SamplingBlocks(const std::vector<std::vector<size_t> >& sampling_blocks)
    {
        sampling_blocks_ = sampling_blocks;

        // make sure sizes are consistent
        size_t dimension = 0;
        for(size_t i = 0; i < sampling_blocks_.size(); i++)
            for(size_t j = 0; j < sampling_blocks_[i].size(); j++)
                dimension++;

        if(dimension != process_model_->NoiseDimension())
        {
            std::cout << "the dimension of the sampling blocks is " << dimension
                      << " while the dimension of the noise is "
                      << process_model_->NoiseDimension() << std::endl;
            exit(-1);
        }
    }

    // get
    const std::vector<State>& Samples() const
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

    std::vector<State > samples_;
    std::vector<size_t> indices_;
    std::vector<Scalar>  log_weights_;
    std::vector<Noise> noises_;
    std::vector<State> next_samples_;
    std::vector<Scalar> loglikes_;

    // observation model
    boost::shared_ptr<ObservationModel> observation_model_;

    // process model
    boost::shared_ptr<ProcessModel> process_model_;

    // parameters
    std::vector<std::vector<size_t> > sampling_blocks_;
    Scalar max_kl_divergence_;

    // distribution for sampling
    Gaussian<Eigen::Matrix<Scalar,1,1> > unit_gaussian_;
};

}

#endif
