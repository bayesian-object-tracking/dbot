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


#include <state_filtering/observation_models/cpu_image_observation_model/cpu_image_observation_model.hpp>

#include <limits>

#include <state_filtering/tools/macros.hpp>
#include <state_filtering/tools/helper_functions.hpp>
#include <state_filtering/tools/image_visualizer.hpp>
//#include "cloud_visualizer.hpp"

using namespace std;
using namespace Eigen;
using namespace obs_mod;


CPUImageObservationModel::CPUImageObservationModel(
        const Eigen::Matrix3d& camera_matrix,
        const size_t& n_rows,
        const size_t& n_cols,
        const size_t& max_sample_count,
        const boost::shared_ptr<RigidBodySystem<-1> >& rigid_body_system,
        const ObjectModel object_model,
        const PixelObservationModel observation_model,
        const OcclusionProcessModel occlusion_process_model,
        const float& initial_visibility_prob):
    ImageObservationModel(camera_matrix, n_rows, n_cols, initial_visibility_prob, max_sample_count, rigid_body_system),
    object_model_(object_model),
    observation_model_(observation_model),
    occlusion_process_model_(occlusion_process_model),
    observation_time_(std::numeric_limits<double>::quiet_NaN())
{
    set_occlusions();
}


CPUImageObservationModel::~CPUImageObservationModel() { }


std::vector<float> CPUImageObservationModel::Evaluate(
        const std::vector<Eigen::VectorXd>& states,
        std::vector<size_t>& occlusion_indices,
        const bool& update_occlusions)
{
    // added for debugging the depth values !!!!!!!!!!!!!!!!!!!!!!!!!
    states_ = states;
    // ------------------------------------

    std::vector<std::vector<float> > new_visibility_probs(states.size());
    std::vector<std::vector<double> > new_visibility_update_times(states.size());
    vector<float> loglikes(states.size(),0);
    for(size_t state_index = 0; state_index < size_t(states.size()); state_index++)
    {


        if(update_occlusions)
        {
            new_visibility_probs[state_index] = visibility_probs_[occlusion_indices[state_index]];
            new_visibility_update_times[state_index] = visibility_update_times_[occlusion_indices[state_index]];
        }
        // we predict observations_ ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::vector<int> intersect_indices;
        std::vector<float> predictions;
        object_model_->set_state(states[state_index]);
        object_model_->PredictObservation(camera_matrix_, n_rows_, n_cols_, intersect_indices, predictions);


        // added for debugging
        //        if (state_index == 0) {
        //            for (int i = 0; i < predictions.size(); i++) {
        //                cout << "(CPU) index: " << intersect_indices[i] << ", depth: " << predictions[i] << endl;
        //            }
        //        }
        // ----------------------------

        // we loop through all the pixels which intersect the object model ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for(size_t i = 0; i < size_t(predictions.size()); i++)
        {
            if(isnan(observations_[intersect_indices[i]]))
                loglikes[state_index] += log(1.);
            else
            {
                float visibility_prob;
                // we predict the visiblity probability and set the time of the last update time to current
                visibility_prob =
                        occlusion_process_model_->Propagate(
                            visibility_probs_[occlusion_indices[state_index]][intersect_indices[i]],
                        observation_time_ - visibility_update_times_[occlusion_indices[state_index]][intersect_indices[i]]);

                float p_obsIpred_vis = // prob of observation given prediction, knowing that the object is visible
                        observation_model_->Prob(observations_[intersect_indices[i]], predictions[i], true)
                        * visibility_prob;
                float p_obsIpred_occl = // prob of observation given prediction, knowing that the object is occluded
                        observation_model_->Prob(observations_[intersect_indices[i]], predictions[i], false)
                        * (1-visibility_prob);
                float p_obsIinf = // prob of observation given no intersection
                        observation_model_->Prob(observations_[intersect_indices[i]],
                        numeric_limits<float>::infinity(), false);

                loglikes[state_index] += log((p_obsIpred_vis + p_obsIpred_occl)/p_obsIinf);

                // we update the visibiliy (occlusion) with the observations
                if(update_occlusions)
                {
                    new_visibility_probs[state_index][intersect_indices[i]] = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
                    new_visibility_update_times[state_index][intersect_indices[i]] = observation_time_;
                }
            }
        }
    }
    if(update_occlusions)
    {
        visibility_probs_ = new_visibility_probs;
        visibility_update_times_ = new_visibility_update_times;
        for(size_t state_index = 0; state_index < occlusion_indices.size(); state_index++)
            occlusion_indices[state_index] = state_index;
    }
    return loglikes;
}




std::vector<float> CPUImageObservationModel::Evaluate_test(
        const std::vector<Eigen::VectorXd>& states,
        std::vector<size_t>& occlusion_indices,
        const bool& update_occlusions,
        vector<vector<int> > intersect_indices,
        vector<vector<float> > predictions)
{
    // added for debugging the depth values
    states_ = states;
    // ------------------------------------

    std::vector<std::vector<float> > new_visibility_probs(states.size());
    std::vector<std::vector<double> > new_visibility_update_times(states.size());
    vector<float> loglikes(states.size(),0);
    for(size_t state_index = 0; state_index < size_t(states.size()); state_index++)
    {

        if(update_occlusions)
        {
            new_visibility_probs[state_index] = visibility_probs_[occlusion_indices[state_index]];
            new_visibility_update_times[state_index] = visibility_update_times_[occlusion_indices[state_index]];
        }

        // we loop through all the pixels which intersect the object model ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for(size_t i = 0; i < size_t(predictions[state_index].size()); i++)
        {
            if(isnan(observations_[intersect_indices[state_index][i]]))
                loglikes[state_index] += log(1.);
            else
            {
                float visibility_prob;
                // we predict the visiblity probability and set the time of the last update time to current
                visibility_prob =
                        occlusion_process_model_->Propagate(visibility_probs_[occlusion_indices[state_index]][intersect_indices[state_index][i]],
                        observation_time_ - visibility_update_times_[occlusion_indices[state_index]][intersect_indices[state_index][i]]);

                //                cout << "time cpu: " << observation_time_ - visibility_update_times_[occlusion_indices[state_index]][intersect_indices[state_index][i]] << endl;

                float p_obsIpred_vis = // prob of observation given prediction, knowing that the object is visible
                        observation_model_->Prob(observations_[intersect_indices[state_index][i]], predictions[state_index][i], true)
                        * visibility_prob;
                float p_obsIpred_occl = // prob of observation given prediction, knowing that the object is occluded
                        observation_model_->Prob(observations_[intersect_indices[state_index][i]], predictions[state_index][i], false)
                        * (1-visibility_prob);
                float p_obsIinf = // prob of observation given no intersection
                        observation_model_->Prob(observations_[intersect_indices[state_index][i]],
                        numeric_limits<float>::infinity(), false);

                loglikes[state_index] += log((p_obsIpred_vis + p_obsIpred_occl)/p_obsIinf);


                // we update the visibiliy (occlusion) with the observations
                if(update_occlusions)
                {
                    new_visibility_probs[state_index][intersect_indices[state_index][i]] = p_obsIpred_vis/(p_obsIpred_vis + p_obsIpred_occl);
                    new_visibility_update_times[state_index][intersect_indices[state_index][i]] = observation_time_;
                }
            }
        }
    }
    if(update_occlusions)
    {
        visibility_probs_ = new_visibility_probs;
        visibility_update_times_ = new_visibility_update_times;
        for(size_t state_index = 0; state_index < occlusion_indices.size(); state_index++)
            occlusion_indices[state_index] = state_index;
    }


    return loglikes;
}




// set and get functions =============================================================================================================================================================================================================================================================================================
const std::vector<float> CPUImageObservationModel::get_occlusions(size_t index) const
{
    return visibility_probs_[index];
}

void CPUImageObservationModel::get_depth_values(std::vector<std::vector<int> > &intersect_indices,
                                                std::vector<std::vector<float> > &depth)
{
    intersect_indices.resize(states_.size());
    depth.resize(states_.size());

    for (size_t i = 0; i < states_.size(); i++) {
        object_model_->set_state(states_[i]);
        object_model_->PredictObservation(camera_matrix_, n_rows_, n_cols_, intersect_indices[i], depth[i]);
    }
}


void CPUImageObservationModel::set_occlusions(const float& visibility_prob)
{
    float p = visibility_prob == -1 ? initial_visibility_prob_ : visibility_prob;
    visibility_probs_ = vector<vector<float> >(1, vector<float>(n_rows_*n_cols_, p));
    visibility_update_times_ = vector<vector<double> >(1, vector<double>(n_rows_*n_cols_, 0));
}


void CPUImageObservationModel::set_observation(const std::vector<float>& observations, const double& observation_time)
{
    observations_ = observations;
    observation_time_ = observation_time;
}
