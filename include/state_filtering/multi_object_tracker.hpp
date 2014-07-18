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

#ifndef MULTI_OBJECT_TRACKER_
#define MULTI_OBJECT_TRACKER_


//#define PROFILING_ON
#include <state_filtering/tools/image_visualizer.hpp>

#include <boost/thread/mutex.hpp>

// ros stuff
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl-1.6/pcl/ros/conversions.h>
#include <pcl-1.6/pcl/point_cloud.h>
#include <pcl-1.6/pcl/point_types.h>
// filter
#include <state_filtering/filter/particle/coordinate_filter.hpp>
#include <state_filtering/filter/particle/particle_filter_context.hpp>

// observation model
#include <state_filtering/observation_models/cpu_image_observation_model/kinect_measurement_model.hpp>
#include <state_filtering/observation_models/image_observation_model.hpp>
#include <state_filtering/observation_models/cpu_image_observation_model/cpu_image_observation_model.hpp>
#include <state_filtering/observation_models/gpu_image_observation_model/gpu_image_observation_model.hpp>

// tools
#include <state_filtering/tools/object_file_reader.hpp>
#include <state_filtering/tools/helper_functions.hpp>
#include <state_filtering/tools/pcl_interface.hpp>
#include <state_filtering/tools/ros_interface.hpp>
#include <state_filtering/tools/macros.hpp>
//#include "cloud_visualizer.hpp"

// distributions
#include <state_filtering/distribution/distribution.hpp>
#include <state_filtering/distribution/implementations/gaussian_distribution.hpp>
#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/process_model/composed_stationary_process_model.hpp>
#include <state_filtering/process_model/brownian_process_model.hpp>

#include <state_filtering/system_states/rigid_body_system.hpp>
#include <state_filtering/system_states/floating_body_system.hpp>

using namespace boost;
using namespace std;
using namespace Eigen;
using namespace filter;





class MultiObjectTracker
{
public:
    typedef filter::ParticleFilterContext<double, -1>   FilterContext;
    typedef boost::shared_ptr<FilterContext>            FilterContextPtr;
    typedef Eigen::Matrix<double, -1, -1> Image;

    MultiObjectTracker():
        node_handle_("~"),
        is_first_iteration_(true),
        duration_(0)
    {
        ri::ReadParameter("object_names", object_names_, node_handle_);
        ri::ReadParameter("downsampling_factor", downsampling_factor_, node_handle_);
        ri::ReadParameter("sample_count", evaluation_count_, node_handle_);

        object_publisher_ = node_handle_.advertise<visualization_msgs::Marker>("object_model", 0);
    }

    void Initialize(
            vector<VectorXd> initial_states,
            const sensor_msgs::Image& ros_image,
            Matrix3d camera_matrix,
            bool state_is_partial = true)
    {
        boost::mutex::scoped_lock lock(mutex_);

        // convert camera matrix and image to desired format ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        camera_matrix.topLeftCorner(2,3) /= double(downsampling_factor_);
        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_); // convert to meters

        // read some parameters ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        bool use_gpu; ri::ReadParameter("use_gpu", use_gpu, node_handle_);

        vector<vector<size_t> > dependencies; ri::ReadParameter("dependencies", dependencies, node_handle_);
        int max_sample_count; ri::ReadParameter("max_sample_count", max_sample_count, node_handle_);

        double p_visible_init; ri::ReadParameter("p_visible_init", p_visible_init, node_handle_);
        double p_visible_visible; ri::ReadParameter("p_visible_visible", p_visible_visible, node_handle_);
        double p_visible_occluded; ri::ReadParameter("p_visible_occluded", p_visible_occluded, node_handle_);

        double linear_acceleration_sigma; ri::ReadParameter("linear_acceleration_sigma", linear_acceleration_sigma, node_handle_);
        double angular_acceleration_sigma; ri::ReadParameter("angular_acceleration_sigma", angular_acceleration_sigma, node_handle_);
        double damping; ri::ReadParameter("damping", damping, node_handle_);

        double tail_weight; ri::ReadParameter("tail_weight", tail_weight, node_handle_);
        double model_sigma; ri::ReadParameter("model_sigma", model_sigma, node_handle_);
        double sigma_factor; ri::ReadParameter("sigma_factor", sigma_factor, node_handle_);

        // initialize observation model ========================================================================================================================================================================================================================================================================================================================================================================================================================
        // load object mesh ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        vector<vector<Vector3d> > object_vertices(object_names_.size());
        vector<vector<vector<int> > > object_triangle_indices(object_names_.size());
        for(size_t i = 0; i < object_names_.size(); i++)
        {
            string object_model_path = ros::package::getPath("arm_object_models") +
                    "/objects/" + object_names_[i] + "/" + object_names_[i] + "_downsampled" + ".obj";
            ObjectFileReader file_reader;
            file_reader.set_filename(object_model_path);
            file_reader.Read();

            object_vertices[i] = *file_reader.get_vertices();
            object_triangle_indices[i] = *file_reader.get_indices();
        } 

        // the rigid_body_system is essentially the state vector with some convenience functions for retrieving
        // the poses of the rigid objects
        boost::shared_ptr<RigidBodySystem<> > rigid_body_system(new FloatingBodySystem<>(object_names_.size()));

        boost::shared_ptr<obj_mod::RigidBodyRenderer> object_renderer(new obj_mod::RigidBodyRenderer(
                                                                          object_vertices,
                                                                          object_triangle_indices,
                                                                          rigid_body_system));

        boost::shared_ptr<obs_mod::ImageObservationModel> observation_model;

        if(!use_gpu)
        {
            // cpu obseration model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            boost::shared_ptr<obs_mod::KinectMeasurementModel>
                    kinect_measurement_model(new obs_mod::KinectMeasurementModel(tail_weight, model_sigma, sigma_factor));
            boost::shared_ptr<proc_mod::OcclusionProcessModel>
                    occlusion_process_model(new proc_mod::OcclusionProcessModel(1. - p_visible_visible, 1. - p_visible_occluded));
            observation_model = boost::shared_ptr<obs_mod::ImageObservationModel>(new obs_mod::CPUImageObservationModel(
                                                                                          camera_matrix,
                                                                                          image.rows(),
                                                                                          image.cols(),
                                                                                          initial_states.size(),
                                                                                          rigid_body_system,
                                                                                          object_renderer,
                                                                                          kinect_measurement_model,
                                                                                          occlusion_process_model,
                                                                                          p_visible_init));
        }
        else
        {
            // gpu obseration model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            boost::shared_ptr<obs_mod::GPUImageObservationModel> gpu_observation_model(new obs_mod::GPUImageObservationModel(
                                                                                           camera_matrix,
                                                                                           image.rows(),
                                                                                           image.cols(),
                                                                                           max_sample_count,
                                                                                           p_visible_init,
                                                                                           rigid_body_system));

            gpu_observation_model->set_constants(object_vertices,
                                                 object_triangle_indices,
                                                 p_visible_visible,
                                                 p_visible_occluded,
                                                 tail_weight,
                                                 model_sigma,
                                                 sigma_factor,
                                                 6.0f,         // max_depth
                                                 -log(0.5));   // exponential_rate

            gpu_observation_model->Initialize();
            observation_model = gpu_observation_model;
        }

        // initialize process model ========================================================================================================================================================================================================================================================================================================================================================================================================================
        MatrixXd linear_acceleration_covariance = MatrixXd::Identity(3, 3) * pow(double(linear_acceleration_sigma), 2);
        MatrixXd angular_acceleration_covariance = MatrixXd::Identity(3, 3) * pow(double(angular_acceleration_sigma), 2);

        vector<boost::shared_ptr<StationaryProcess<> > > partial_process_models(object_names_.size());
        for(size_t i = 0; i < partial_process_models.size(); i++)
        {
            boost::shared_ptr<BrownianProcessModel<> > partial_process_model(new BrownianProcessModel<>);
            partial_process_model->parameters(object_renderer->object_center(i).cast<double>(),
                                              damping,
                                              linear_acceleration_covariance,
                                              angular_acceleration_covariance);
            partial_process_models[i] = partial_process_model;
        }
        boost::shared_ptr<ComposedStationaryProcessModel> process_model(new ComposedStationaryProcessModel(partial_process_models));

        // initialize coordinate_filter ============================================================================================================================================================================================================================================================
        filter_ = boost::shared_ptr<filter::CoordinateFilter>
                (new filter::CoordinateFilter(observation_model, process_model, dependencies));


        if(state_is_partial)
        {
            // create the multi body initial samples ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            FloatingBodySystem<> default_state(object_names_.size());
            for(size_t object_index = 0; object_index < object_names_.size(); object_index++)
                default_state.position(object_index) = Vector3d(0, 0, 1.5); // outside of image

            cout << "creating intiial stuff" << endl;
            vector<VectorXd> multi_body_samples(initial_states.size());
            for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
                multi_body_samples[state_index] = default_state;

            cout << "doing evaluations " << endl;
            for(size_t body_index = 0; body_index < object_names_.size(); body_index++)
            {
                cout << "evalution of object " << object_names_[body_index] << endl;
                for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
                {
                    FloatingBodySystem<> full_initial_state(multi_body_samples[state_index]);
                    full_initial_state[body_index] = initial_states[state_index];
                    multi_body_samples[state_index] = full_initial_state;
                }
                filter_->set_states(multi_body_samples);
                filter_->Evaluate(image);
                filter_->Resample(multi_body_samples.size());
                filter_->get(multi_body_samples);
            }

            // we evaluate the initial particles and resample ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cout << "evaluating initial particles cpu ..." << endl;
            filter_->set_states(multi_body_samples);
            filter_->Evaluate(image);
            filter_->Resample(evaluation_count_/(dependencies.size() * 10)); // FOR TESTING ONLY
        }
        else
        {
            filter_->set_states(initial_states);
            filter_->Evaluate(image);
            filter_->Resample(evaluation_count_/(dependencies.size() * 10));// FOR TESTING ONLY
        }
    }

    VectorXd Filter(const sensor_msgs::Image& ros_image)
    {
        boost::mutex::scoped_lock lock(mutex_);
        // the time since start is computed

        if(is_first_iteration_)
        {
            previous_image_time_ = ros_image.header.stamp.toSec();
            is_first_iteration_ = false;
        }
        duration_ += ros_image.header.stamp.toSec() - previous_image_time_;

        // convert image
        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_); // convert to m

        // filter
        INIT_PROFILING;
//        filter_->Enchilada(VectorXd::Zero(object_names_.size()*6),
//                           duration_,
//                           image,
//                           evaluation_count_);

        filter_->Enchiladisima(VectorXd::Zero(object_names_.size()*6),
                               duration_,
                               image,
                               evaluation_count_);
        MEASURE("-----------------> total time for filtering");

        previous_image_time_ = ros_image.header.stamp.toSec();

        // visualize the mean state
        FloatingBodySystem<> mean = filter_->stateDistribution().empiricalMean();
        for(size_t i = 0; i < object_names_.size(); i++)
        {
            string object_model_path = "package://arm_object_models/objects/" + object_names_[i] + "/" + object_names_[i] + ".obj";
            ri::PublishMarker(mean.homogeneous_matrix(i).cast<float>(),
                              ros_image.header, object_model_path, object_publisher_,
                              i, 1, 0, 0);
        }

        PRINT("sample count ") PRINT(evaluation_count_)

        return filter_->stateDistribution().empiricalMean();
    }




private:  
    double duration_;



    boost::mutex mutex_;
    ros::NodeHandle node_handle_;
    ros::Publisher object_publisher_;

    boost::shared_ptr<filter::CoordinateFilter> filter_;

    bool is_first_iteration_;
    double previous_image_time_;

    // parameters
    vector<string> object_names_;
    int downsampling_factor_;
    int evaluation_count_;
};

#endif

