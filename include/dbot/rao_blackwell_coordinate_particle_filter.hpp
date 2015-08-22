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

#ifndef DBOT__RB_COORDINATE_PARTICLE_FILTER_HPP
#define DBOT__RB_COORDINATE_PARTICLE_FILTER_HPP

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

    typedef Eigen::Array<State, -1, 1>       StateArray;
    typedef Eigen::Array<fl::Real, -1, 1>    RealArray;
    typedef Eigen::Array<int, -1, 1>         IntArray;


    typedef typename ObservationModel::Observation Observation;

    typedef fl::DiscreteDistribution<State> Belief;

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

        // make sure sizes are consistent --------------------------------------
        size_t dimension = 0;
        for(size_t i = 0; i < sampling_blocks_.size(); i++)
        {
           dimension +=  sampling_blocks_[i].size();
        }
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
        observation_model_->set_observation(observation);

        loglikes_ = RealArray::Zero(belief_.size());
        noises_ = std::vector<Noise>(belief_.size(),
                                 Noise::Zero(process_model_->NoiseDimension()));
        old_particles_ = belief_.locations();

        for(size_t i_block = 0; i_block < sampling_blocks_.size(); i_block++)
        {
            // add noise of this block -----------------------------------------
            for(size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                for(size_t i = 0; i < sampling_blocks_[i_block].size(); i++)
                {
                    noises_[i_sampl](sampling_blocks_[i_block][i]) =
                                                    unit_gaussian_.sample()(0);
                }
            }

            // propagate using partial noise -----------------------------------
            for(size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                process_model_->Condition(old_particles_[i_sampl], input);
                belief_.location(i_sampl) =
                        process_model_->MapStandardGaussian(noises_[i_sampl]);
            }

            // compute likelihood ----------------------------------------------
            bool update = (i_block == sampling_blocks_.size()-1);
            RealArray new_loglikes =
            observation_model_->loglikes(belief_.locations(), indices_, update);

            // update the weights and resample if necessary --------------------
            belief_.delta_log_prob_mass(new_loglikes - loglikes_);
            loglikes_ = new_loglikes;

            if(belief_.kl_given_uniform() > max_kl_divergence_)
            {
                resample(belief_.size());
            }
        }
    }

    void resample(const size_t& sample_count)
    {
        IntArray indices(sample_count);
        std::vector<Noise> noises(sample_count);
        StateArray next_samples(sample_count);
        RealArray loglikes(sample_count);

        Belief new_belief(sample_count);

        for(size_t i = 0; i < sample_count; i++)
        {
            int index;
            new_belief.location(i) = belief_.sample(index);

            indices[i]      = indices_[index];
            noises[i]       = noises_[index];
            next_samples[i] = old_particles_[index];
            loglikes[i]     = loglikes_[index];
        }
        belief_         = new_belief;
        indices_        = indices;
        noises_         = noises;
        old_particles_   = next_samples;
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

        indices_ = IntArray::Zero(samples.size());
        observation_model_->reset();
    }


private:
    /// member variables *******************************************************
    Belief belief_;
    IntArray indices_;

    std::vector<Noise> noises_;
    StateArray old_particles_;
    RealArray loglikes_;

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
