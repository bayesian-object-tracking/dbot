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

#include <state_filtering/models/measurement/image_observation_model.hpp>
//#include <state_filtering/models/measurement/cpu_image_observation_modegaussian_pixel_observation_modelel.hpp>
//#include <state_filtering/models/process/implementations/occlusion_process.hpp>

#include <state_filtering/utils/rigid_body_renderer.hpp>
#include <state_filtering/models/process/features/stationary_process.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>
#include <state_filtering/distributions/implementations/sum_of_deltas.hpp>

#include <state_filtering/states/floating_body_system.hpp>

#include <state_filtering/models/process/implementations/brownian_object_motion.hpp>


/// this namespace contains all the filters
namespace distributions
{

class CoordinateParticleFilter
{
public:
    typedef double ScalarType;
    typedef FloatingBodySystem<-1> VectorType;
    typedef Eigen::VectorXd Control;
    typedef Eigen::VectorXd Noise;




    typedef boost::shared_ptr<CoordinateParticleFilter> Ptr;

    typedef distributions::BrownianObjectMotion<-1, double> ProcessModel;
    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;
    typedef obs_mod::ImageObservationModel MeasurementModel;
    typedef boost::shared_ptr<MeasurementModel> MeasurementModelPtr;

    typedef SumOfDeltas<double, Eigen::VectorXd> StateDistribution;

    typedef MeasurementModel::Measurement Measurement;
    typedef Eigen::VectorXd State;

    CoordinateParticleFilter(const MeasurementModelPtr observation_model,
                     const ProcessModelPtr process_model,
                     const std::vector<std::vector<size_t> >& independent_blocks,
                     const double& max_kl_divergence = 0);

    ~CoordinateParticleFilter();

    void Enchilada(const Control control,
                   const double &observation_time,
                   const Measurement& observation,
                   const size_t &evaluation_count);


    void Enchiladisima(const Control control,
                   const double &observation_time,
                   const Measurement& observation,
                   const size_t &evaluation_count,
                   const size_t &factor_evaluation_count);


    void Filter(const Control control,
                   const double &observation_time,
                   const Measurement& observation);


    void Propagate(const Control control,
                   const double &current_time);


    void Evaluate(const Measurement& observation,
                  const double& observation_time = std::numeric_limits<double>::quiet_NaN(),
                  const bool& update_occlusions = false);


    void UpdateWeights(std::vector<float> log_weight_diffs);


    void Resample(const int &new_state_count = -1);

    size_t control_size();

    // set and get fcts ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    void get(MeasurementModelPtr& observation_model) const;
    void get(ProcessModelPtr& process_model) const;
    void get(std::vector<State >& states) const;
    void get(std::vector<double>& state_times) const;
    void get(std::vector<float>& loglikes) const;
    const State& get_state(size_t index) const;
    const std::vector<float> get_occlusions(size_t index) const;
    void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                          std::vector<std::vector<float> > &depth);

    void set_states(const std::vector<State >& states,
                    const std::vector<double>& state_times = std::vector<double>(),
                    const std::vector<float>& loglikes = std::vector<float>());
    void set_independence(const std::vector<std::vector<size_t> >& independent_blocks);
    void set(const MeasurementModelPtr& observation_model);
    void set(const ProcessModelPtr& process_model);

    virtual StateDistribution& stateDistribution();

private:

    unsigned dimension_;

    // TODO this is not used properly yet
    StateDistribution state_distribution_;


    // internal state ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<State > particles_;
    std::vector<double> particle_times_;
    std::vector<size_t> occlusion_indices_;
    std::vector<float>  log_weights_;

    std::vector<Noise> noises_;
    std::vector<State> propagated_particles_;
    std::vector<float> loglikes_;




    // partial evaluate
    std::vector< std::vector< std::vector <float> > > partial_children_loglikes_;
    std::vector<float> family_loglikes_;

    // process and observation models ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    MeasurementModelPtr measurement_model_;
    ProcessModelPtr process_model_;
    Gaussian<double, 1> unit_gaussian_;

    // parameters ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<std::vector<size_t> > independent_blocks_;


    const double max_kl_divergence_;
};

}

#endif
