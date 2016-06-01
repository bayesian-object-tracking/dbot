/*
 * This is part of the Bayesian Object Tracking (bot),
 * (https://github.com/bayesian-object-tracking)
 *
 * Copyright (c) 2015 Max Planck Society,
 * 				 Autonomous Motion Department,
 * 			     Institute for Intelligent Systems
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License License (GNU GPL). A copy of the license can be found in the LICENSE
 * file distributed with this source code.
 */

/*
 * This file implements a part of the algorithm published in:
 *
 * M. Wuthrich, J. Bohg, D. Kappler, C. Pfreundt, S. Schaal
 * The Coordinate Particle Filter -
 * A novel Particle Filter for High Dimensional Systems
 * IEEE Intl Conf on Robotics and Automation, 2015
 * http://arxiv.org/abs/1505.00251
 *
 */

#pragma once

#include <vector>
#include <limits>
#include <string>
#include <memory>

#include <Eigen/Core>

#include <fl/util/types.hpp>
#include <fl/distribution/gaussian.hpp>
#include <fl/distribution/discrete_distribution.hpp>
#include <fl/util/profiling.hpp>

#include <dbot/util/traits.hpp>
#include <dbot/model/observation/rao_blackwell_observation_model.hpp>

namespace dbot
{
template <typename ProcessModel, typename ObservationModel>
class RaoBlackwellCoordinateParticleFilter
{
public:
    typedef typename ProcessModel::State State;
    typedef typename ProcessModel::Input Input;
    typedef typename ProcessModel::Noise Noise;

    typedef Eigen::Array<State, -1, 1> StateArray;
    typedef Eigen::Array<fl::Real, -1, 1> RealArray;
    typedef Eigen::Array<int, -1, 1> IntArray;

    typedef typename ObservationModel::Observation Observation;

    typedef fl::DiscreteDistribution<State> Belief;

public:
    /// constructor and destructor *********************************************
    RaoBlackwellCoordinateParticleFilter(
        const std::shared_ptr<ProcessModel> process_model,
        const std::shared_ptr<ObservationModel> observation_model,
        const std::vector<std::vector<int>>& sampling_blocks,
        const fl::Real& max_kl_divergence = 0)
        : observation_model_(observation_model),
          process_model_(process_model),
          max_kl_divergence_(max_kl_divergence)
    {
        sampling_blocks_ = sampling_blocks;

        // make sure sizes are consistent --------------------------------------
        size_t dimension = 0;
        for (size_t i = 0; i < sampling_blocks_.size(); i++)
        {
            dimension += sampling_blocks_[i].size();
        }
        if (dimension != process_model_->noise_dimension())
        {
            std::cout << "the dimension of the sampling blocks is " << dimension
                      << " while the dimension of the noise is "
                      << process_model_->noise_dimension() << std::endl;
            exit(-1);
        }
    }
    virtual ~RaoBlackwellCoordinateParticleFilter() noexcept {}
    /// the filter functions ***************************************************
    void filter(const Observation& observation, const Input& input)
    {
        observation_model_->set_observation(observation);

        loglikes_ = RealArray::Zero(belief_.size());
        noises_ = std::vector<Noise>(
            belief_.size(), Noise::Zero(process_model_->noise_dimension()));
        old_particles_ = belief_.locations();
        for (size_t i_block = 0; i_block < sampling_blocks_.size(); i_block++)
        {
            // add noise of this block -----------------------------------------
            for (size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                for (size_t i = 0; i < sampling_blocks_[i_block].size(); i++)
                {
                    noises_[i_sampl](sampling_blocks_[i_block][i]) =
                        unit_gaussian_.sample()(0);
                }
            }

            // propagate using partial noise -----------------------------------
            for (size_t i_sampl = 0; i_sampl < belief_.size(); i_sampl++)
            {
                belief_.location(i_sampl) = process_model_->state(
                    old_particles_[i_sampl], noises_[i_sampl], input);
            }

            // compute likelihood ----------------------------------------------
            bool update = (i_block == sampling_blocks_.size() - 1);
            RealArray new_loglikes = observation_model_->loglikes(
                belief_.locations(), indices_, update);

            // update the weights and resample if necessary --------------------
            belief_.delta_log_prob_mass(new_loglikes - loglikes_);
            loglikes_ = new_loglikes;

            if (belief_.kl_given_uniform() > max_kl_divergence_)
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

        for (size_t i = 0; i < sample_count; i++)
        {
            int index;
            new_belief.location(i) = belief_.sample(index);

            indices[i] = indices_[index];
            noises[i] = noises_[index];
            next_samples[i] = old_particles_[index];
            loglikes[i] = loglikes_[index];
        }
        belief_ = new_belief;
        indices_ = indices;
        noises_ = noises;
        old_particles_ = next_samples;
        loglikes_ = loglikes;
    }

    /// mutators ***************************************************************
    Belief& belief() { return belief_; }
    void set_particles(const std::vector<State>& samples)
    {
        belief_.set_uniform(samples.size());
        for (int i = 0; i < belief_.size(); i++)
            belief_.location(i) = samples[i];

        indices_ = IntArray::Zero(belief_.size());
        loglikes_ = RealArray::Zero(belief_.size());
        noises_ = std::vector<Noise>(
            belief_.size(), Noise::Zero(process_model_->noise_dimension()));
        old_particles_ = belief_.locations();

        observation_model_->reset();
    }

    std::shared_ptr<ObservationModel> observation_model()
    {
        return observation_model_;
    }

    std::shared_ptr<ProcessModel> process_model()
    {
        return process_model_;
    }

private:
    /// member variables *******************************************************
    Belief belief_;
    IntArray indices_;

    std::vector<Noise> noises_;
    StateArray old_particles_;
    RealArray loglikes_;

    // models
    std::shared_ptr<ObservationModel> observation_model_;
    std::shared_ptr<ProcessModel> process_model_;

    // parameters
    std::vector<std::vector<int>> sampling_blocks_;
    fl::Real max_kl_divergence_;

    // distribution for sampling
    fl::Gaussian<Eigen::Matrix<fl::Real, 1, 1>> unit_gaussian_;
};
}
