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

#include <state_filtering/observation_models/image_observation_model.hpp>
//#include <state_filtering/observation_models/cpu_image_observation_modegaussian_pixel_observation_modelel.hpp>
//#include <state_filtering/process_model/occlusion_process_model.hpp>

#include <state_filtering/tools/rigid_body_renderer.hpp>
#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>
#include <state_filtering/distribution/implementations/sum_of_deltas.hpp>


/// this namespace contains all the filters
namespace filter
{

class CoordinateFilter
{
public:
    typedef StationaryProcessModel< > ProcessModel;
    typedef obs_mod::ImageObservationModel MeasurementModel;
    typedef SumOfDeltas<double, -1> StateDistribution;

    typedef boost::shared_ptr<CoordinateFilter> Ptr;
    typedef boost::shared_ptr<MeasurementModel> MeasurementModelPtr;
    typedef boost::shared_ptr<ProcessModel> ProcessModelPtr;

    typedef std::vector<float> Measurement;

    CoordinateFilter(const MeasurementModelPtr observation_model,
                     const ProcessModelPtr process_model,
                     const std::vector<std::vector<size_t> >& independent_blocks);

    ~CoordinateFilter();

    void PartialPropagate(const Eigen::VectorXd& control,
                          const double& observation_time);
    void PartialEvaluate(const Measurement& observation,
                         const double& observation_time);
    void PartialResample(const Eigen::VectorXd& control,
                         const double& observation_time,
                         const size_t &new_n_states);
    void UpdateOcclusions(const Measurement& observation,
                          const double& observation_time);


    void Enchilada(const Eigen::VectorXd control,
                   const double &observation_time,
                   const Measurement& observation,
                   const size_t &new_n_states);


    void Propagate(const Eigen::VectorXd control,
                   const double &current_time);


    void Evaluate(const Measurement& observation,
                  const double& observation_time = std::numeric_limits<double>::quiet_NaN(),
                  const bool& update_occlusions = false);

    void Resample(const int &new_state_count = -1);

    void Sort();

    // set and get fcts ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    void get(MeasurementModelPtr& observation_model) const;
    void get(ProcessModelPtr& process_model) const;
    void get(std::vector<Eigen::VectorXd >& states) const;
    void get(std::vector<double>& state_times) const;
    void get(std::vector<size_t>& multiplicities) const;
    void get(std::vector<float>& loglikes) const;
    const Eigen::VectorXd& get_state(size_t index) const;
    const std::vector<float> get_occlusions(size_t index) const;
    void get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                          std::vector<std::vector<float> > &depth);

    void set_states(const std::vector<Eigen::VectorXd >& states,
                    const std::vector<double>& state_times = std::vector<double>(),
                    const std::vector<size_t>& multiplicities = std::vector<size_t>(),
                    const std::vector<float>& loglikes = std::vector<float>());
    void set_independence(const std::vector<std::vector<size_t> >& independent_blocks);
    void set(const MeasurementModelPtr& observation_model);
    void set(const ProcessModelPtr& process_model);

    virtual StateDistribution& stateDistribution();

private:
    // TODO this is not used properly yet
    StateDistribution state_distribution_;


    // internal state ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<Eigen::VectorXd > parents_;
    std::vector<double> parent_times_;
    std::vector<size_t> parent_multiplicities_;
    std::vector<size_t> parent_occlusion_indices_;

    // partial propagate
    std::vector<std::vector<std::vector<Eigen::VectorXd> > > partial_noises_;
    std::vector<std::vector<std::vector<Eigen::VectorXd> > > partial_children_;
    std::vector<std::vector<std::vector<size_t> > > partial_children_occlusion_indices_;
    std::vector<Eigen::VectorXd > zero_children_;

    // partial evaluate
    std::vector< std::vector< std::vector <float> > > partial_children_loglikes_;
    std::vector<float> family_loglikes_;

    // process and observation models ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    MeasurementModelPtr observation_model_;
    ProcessModelPtr process_model_;
    GaussianDistribution<double, 1> unit_gaussian_;

    // parameters ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<std::vector<size_t> > independent_blocks_;
    size_t dof_count_;
};

}

#endif
