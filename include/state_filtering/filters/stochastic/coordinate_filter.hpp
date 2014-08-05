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


/// this namespace contains all the filters
namespace distributions
{

class CoordinateParticleFilter
{
public:
    typedef double ScalarType;
    typedef size_t IndexType;
    typedef FloatingBodySystem<-1> StateType;



    typedef Eigen::VectorXd InputType;


    typedef Eigen::Matrix<ScalarType, -1, -1>   MeasurementType;


    typedef distributions::BrownianObjectMotion<-1, ScalarType> ProcessModel;

    typedef distributions::RaoBlackwellMeasurementModel<ScalarType, StateType, MeasurementType> MeasurementModel;


    typedef SumOfDeltas<ScalarType, StateType> StateDistributionType;

    typedef MeasurementModel::MeasurementType Measurement;



    typedef Eigen::VectorXd Noise;




    typedef boost::shared_ptr<CoordinateParticleFilter> Ptr;

    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;
    typedef boost::shared_ptr<MeasurementModel> MeasurementModelPtr;


    CoordinateParticleFilter(const MeasurementModelPtr observation_model,
                     const ProcessModelPtr process_model,
                     const std::vector<std::vector<IndexType> >& independent_blocks,
                     const double& max_kl_divergence = 0);

    ~CoordinateParticleFilter();



    void Filter(const Measurement& measurement,
                const ScalarType&  delta_time,
                const InputType&   input);

    void Evaluate(const Measurement& observation,
                  const double& delta_time = 0,
                  const bool& update_occlusions = false);




    void Resample(const int &new_state_count = -1);

    size_t control_size();

    // set and get fcts ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//    void get(MeasurementModelPtr& observation_model) const;
//    void get(ProcessModelPtr& process_model) const;
//    void get(std::vector<float>& loglikes) const;
//    const StateType& get_state(size_t index) const;



    const std::vector<StateType>& Samples() const;
    void Samples(const std::vector<StateType >& states);


    virtual StateDistributionType& StateDistribution();

private:

    void UpdateWeights(std::vector<float> log_weight_diffs);


    unsigned dimension_;

    // TODO this is not used properly yet
    StateDistributionType state_distribution_;


    // internal state ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<StateType > samples_;
    std::vector<IndexType> indices_;
    std::vector<float>  log_weights_;

    std::vector<Noise> noises_;
    std::vector<StateType> propagated_samples_;
    std::vector<float> loglikes_;




    // TODO: THIS SHOULD GO AWAY
    std::vector<float> family_loglikes_;

    // process and observation models
    MeasurementModelPtr measurement_model_;
    ProcessModelPtr process_model_;

    // distribution for sampling
    Gaussian<double, 1> unit_gaussian_;

    // parameters
    std::vector<std::vector<size_t> > independent_blocks_;
    const double max_kl_divergence_;
};

}

#endif
