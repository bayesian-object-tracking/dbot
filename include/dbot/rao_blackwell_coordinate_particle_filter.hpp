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

#ifndef FAST_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP
#define FAST_FILTERING_FILTERS_STOCHASTIC_RAO_BLACKWELL_COORDINATE_PARTICLE_FILTER_HPP

#include <vector>
#include <limits>
#include <string>

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <fl/util/types.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/discrete_distribution.hpp>

#include <dbot/utils/profiling.hpp>
#include <dbot/utils/traits.hpp>
#include <dbot/utils/helper_functions.hpp>
#include <dbot/models/observation_models/rao_blackwell_observation_model.hpp>



namespace ff
{

template<typename ProcessModel, typename ObservationModel>
class RBCoordinateParticleFilter
{
public:
    typedef typename internal::Traits<ProcessModel>::State  State;
    typedef typename internal::Traits<ProcessModel>::Input  Input;
    typedef typename internal::Traits<ProcessModel>::Noise  Noise;


    typedef typename ObservationModel::Observation Observation;

    typedef fl::DiscreteDistribution<State> Belief;

    typedef typename Belief::Function List;

public:
    /// constructor and destructor *********************************************
    RBCoordinateParticleFilter(
            const boost::shared_ptr<ProcessModel>       process_model,
            const boost::shared_ptr<ObservationModel>   observation_model,
            const std::vector<std::vector<size_t> >&    sampling_blocks,
            const fl::Real&                             max_kl_divergence = 0):
        observation_model_(observation_model),
        process_model_(process_model),
        max_kl_divergence_(max_kl_divergence)
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
    virtual ~RBCoordinateParticleFilter() {}

    /// the filter functions ***************************************************
    void filter(const Observation&  observation,
                const Input&        input)
    {
        observation_model_->SetObservation(observation);

        loglikes_ = List::Zero(belief_.size());
        noises_ = std::vector<Noise>(belief_.size(),
                                 Noise::Zero(process_model_->NoiseDimension()));

        next_samples_.resize(belief_.size());
        for(int i = 0; i < belief_.size(); i++)
            next_samples_[i] = belief_.location(i);

        for(size_t i_block = 0; i_block < sampling_blocks_.size(); i_block++)
        {
            INIT_PROFILING;
            // add noise of this block -----------------------------------------
            for(size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                for(size_t i = 0; i < sampling_blocks_[i_block].size(); i++)
                {
                    noises_[i_sampl](sampling_blocks_[i_block][i]) =
                                                    unit_gaussian_.sample()(0);
                }
            }
            MEASURE("sampling");

            // propagate using partial noise -----------------------------------
            for(size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                process_model_->Condition(belief_.location(i_sampl), input);
                next_samples_[i_sampl] =
                        process_model_->MapStandardGaussian(noises_[i_sampl]);
            }
            MEASURE("propagation");

            // compute likelihood ----------------------------------------------
            bool update_occlusions = (i_block == sampling_blocks_.size()-1);
            std::vector<fl::Real> new_loglikes_std =
                    observation_model_->Loglikes(next_samples_,
                                                 indices_, update_occlusions);

            List new_loglikes(new_loglikes_std.size());
            for(size_t i = 0; i < new_loglikes.size(); i++)
            {
                new_loglikes[i] = new_loglikes_std[i];
            }
            MEASURE("evaluation");

            // update the weights and resample if necessary --------------------
            belief_.delta_log_prob_mass(new_loglikes - loglikes_);

            if(belief_.kl_given_uniform() > max_kl_divergence_)
            {
                resample(belief_.size());
            }
            loglikes_ = new_loglikes;
            MEASURE("updating weights");
        }

        for(int i = 0; i < belief_.size(); i++)
        {
            belief_.location(i) = next_samples_[i];
        }
    }

    void resample(const size_t& sample_count)
    {
        std::vector<size_t> indices(sample_count);
        std::vector<Noise> noises(sample_count);
        std::vector<State> next_samples(sample_count);
        List loglikes(sample_count);

        Belief new_belief(sample_count);

        for(size_t i = 0; i < sample_count; i++)
        {
            int index;
            new_belief.location(i) = belief_.sample(index);

            indices[i]      = indices_[index];
            noises[i]       = noises_[index];
            next_samples[i] = next_samples_[index];
            loglikes[i]     = loglikes_[index];
        }
        belief_         = new_belief;
        indices_        = indices;
        noises_         = noises;
        next_samples_   = next_samples;
        loglikes_       = loglikes;
    }


    /// mutators ***************************************************************
    Belief& belief()
    {
        return belief_;
    }

    void set_particles(const std::vector<State >& samples)
    {
        belief_.set_uniform(samples.size());
        for(int i = 0; i < belief_.size(); i++)
            belief_.location(i) = samples[i];

        indices_ = std::vector<size_t>(samples.size(), 0);
        observation_model_->Reset();
    }


private:
    Belief belief_;
    std::vector<size_t> indices_;

    std::vector<Noise> noises_;
    std::vector<State> next_samples_;
    List loglikes_;

    // models
    boost::shared_ptr<ObservationModel> observation_model_;
    boost::shared_ptr<ProcessModel>     process_model_;

    // parameters
    std::vector<std::vector<size_t> > sampling_blocks_;
    fl::Real max_kl_divergence_;

    // distribution for sampling
    fl::Gaussian<Eigen::Matrix<fl::Real,1,1> > unit_gaussian_;
};

}

#endif
