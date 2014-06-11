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
#include <state_filtering/system_states/full_rigid_body_system.hpp>

using namespace boost;
using namespace std;
using namespace Eigen;
using namespace filter;





class RunCpuCoordinateFilter
{
public:
    typedef filter::ParticleFilterContext<double, -1>   FilterContext;
    typedef boost::shared_ptr<FilterContext>            FilterContextPtr;
    typedef Eigen::Matrix<double, -1, -1> Image;

    RunCpuCoordinateFilter(Matrix3d camera_matrix):
        node_handle_("~"),
        is_first_iteration_(true)
    {
        // read parameters ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ri::ReadParameter("dependencies", dependencies_, node_handle_);
        ri::ReadParameter("object_names", object_names_, node_handle_);
        ri::ReadParameter("downsampling_factor", downsampling_factor_, node_handle_);
        ri::ReadParameter("cpu_sample_count", cpu_sample_count_, node_handle_);
        ri::ReadParameter("p_visible_init", p_visible_init_, node_handle_);
        ri::ReadParameter("p_visible_visible", p_visible_visible_, node_handle_);
        ri::ReadParameter("p_visible_occluded", p_visible_occluded_, node_handle_);

        camera_matrix_ = camera_matrix;
        camera_matrix_.topLeftCorner(2,3) /= float(downsampling_factor_);
        object_publisher_ = node_handle_.advertise<visualization_msgs::Marker>("object_model", 0);
    }

    void Initialize(
            vector<VectorXd> single_body_samples,
            const sensor_msgs::Image& ros_image)
    {
        boost::mutex::scoped_lock lock(mutex_);

        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_) / 1000.; // convert to m

        vector<float> measurement(image.size()); size_t n_rows = image.rows(), n_cols = image.cols();
        
        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                measurement[row*image.cols() + col] = image(row, col);
        
        // initialize observation model ========================================================================================================================================================================================================================================================================================================================================================================================================================
        // load object mesh ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        boost::shared_ptr<RigidBodySystem<> > rigid_body_system(new FullRigidBodySystem<>(object_names_.size()));

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

        boost::shared_ptr<obj_mod::RigidBodyRenderer> object_renderer(new obj_mod::RigidBodyRenderer(
                                                                       object_vertices,
                                                                       object_triangle_indices,
                                                                       rigid_body_system));
        // kinect_measurement_model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        double tail_weight; ri::ReadParameter("tail_weight", tail_weight, node_handle_);
        double model_sigma; ri::ReadParameter("model_sigma", model_sigma, node_handle_);
        double sigma_factor; ri::ReadParameter("sigma_factor", sigma_factor, node_handle_);
        boost::shared_ptr<obs_mod::KinectMeasurementModel>
                kinect_measurement_model(new obs_mod::KinectMeasurementModel(tail_weight, model_sigma, sigma_factor));

        // initialize occlusion process model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        boost::shared_ptr<proc_mod::OcclusionProcessModel>
                occlusion_process_model(new proc_mod::OcclusionProcessModel(1. - p_visible_visible_, 1. - p_visible_occluded_));

        // cpu obseration model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        boost::shared_ptr<obs_mod::ImageObservationModel> cpu_observation_model(new obs_mod::CPUImageObservationModel(
                                                                                    camera_matrix_,
                                                                                    n_rows,
                                                                                    n_cols,
                                                                                    single_body_samples.size(),
                                                                                    rigid_body_system,
                                                                                    object_renderer,
                                                                                    kinect_measurement_model,
                                                                                    occlusion_process_model,
                                                                                    p_visible_init_));

        // initialize process model ========================================================================================================================================================================================================================================================================================================================================================================================================================
        double free_damping; ri::ReadParameter("free_damping", free_damping, node_handle_);

        double free_linear_acceleration_sigma; ri::ReadParameter("free_linear_acceleration_sigma", free_linear_acceleration_sigma, node_handle_);
        MatrixXd free_linear_acceleration_covariance =
                MatrixXd::Identity(3, 3) * pow(double(free_linear_acceleration_sigma), 2);

        double free_angular_acceleration_sigma; ri::ReadParameter("free_angular_acceleration_sigma", free_angular_acceleration_sigma, node_handle_);
        MatrixXd free_angular_acceleration_covariance =
                MatrixXd::Identity(3, 3) * pow(double(free_angular_acceleration_sigma), 2);

        vector<boost::shared_ptr<StationaryProcessModel<> > > partial_process_models(object_names_.size());
        for(size_t i = 0; i < partial_process_models.size(); i++)
        {
            boost::shared_ptr<BrownianProcessModel<> > partial_process_model(new BrownianProcessModel<>);
            partial_process_model->parameters(object_renderer->object_center(i).cast<double>(),
                                              free_damping,
                                              free_linear_acceleration_covariance,
                                              free_angular_acceleration_covariance);
            partial_process_models[i] = partial_process_model;
        }

        boost::shared_ptr<ComposedStationaryProcessModel> process_model
                (new ComposedStationaryProcessModel(partial_process_models));


        // initialize coordinate_filter ============================================================================================================================================================================================================================================================
        cpu_filter_ = boost::shared_ptr<filter::CoordinateFilter>
                (new filter::CoordinateFilter(cpu_observation_model, process_model, dependencies_));


        // create the multi body initial samples ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        FullRigidBodySystem<> default_state(object_names_.size());
        for(size_t object_index = 0; object_index < object_names_.size(); object_index++)
            default_state.position(object_index) = Vector3d(0, 0, 1.5); // outside of image

        cout << "creating intiial stuff" << endl;
        vector<VectorXd> multi_body_samples(single_body_samples.size());
        for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
            multi_body_samples[state_index] = default_state;

        cout << "doing evaluations " << endl;
        for(size_t body_index = 0; body_index < object_names_.size(); body_index++)
        {
            cout << "evalution of object " << object_names_[body_index] << endl;
            for(size_t state_index = 0; state_index < multi_body_samples.size(); state_index++)
            {
                FullRigidBodySystem<> full_initial_state(multi_body_samples[state_index]);
                full_initial_state[body_index] = single_body_samples[state_index];
                multi_body_samples[state_index] = full_initial_state;
            }
            cpu_filter_->set_states(multi_body_samples);
            cpu_filter_->Evaluate(measurement);
            cpu_filter_->Resample(multi_body_samples.size());
            cpu_filter_->get(multi_body_samples);
        }

        // we evaluate the initial particles and resample ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cout << "evaluating initial particles cpu ..." << endl;
        cpu_filter_->set_states(multi_body_samples);
        cpu_filter_->Evaluate(measurement);
        cpu_filter_->Resample(cpu_sample_count_);

        filter_context_ =
                boost::shared_ptr<filter::ParticleFilterContext<double, -1> >
                (new filter::ParticleFilterContext<double, -1>(cpu_filter_) );
    }






    void Filter(const sensor_msgs::Image& ros_image)
    {
        boost::mutex::scoped_lock lock(mutex_);

        // the time since start is computed ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if(is_first_iteration_)
        {
            previous_time_ = ros_image.header.stamp.toSec();
            is_first_iteration_ = false;
        }


        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_) / 1000.; // convert to m

        vector<float> measurement(image.size()); size_t n_rows = image.rows(), n_cols = image.cols();

        for(size_t row = 0; row < image.rows(); row++)
            for(size_t col = 0; col < image.cols(); col++)
                measurement[row*image.cols() + col] = image(row, col);

        // filter: this is where stuff happens ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        INIT_PROFILING;
        filter_context_->predictAndUpdate(measurement,
                                          ros_image.header.stamp.toSec() - previous_time_,
                                          VectorXd::Zero(object_names_.size()*6));
        MEASURE("-----------------> total time for filtering");

        previous_time_ = ros_image.header.stamp.toSec();

        // we visualize the likeliest state -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        FullRigidBodySystem<> mean = filter_context_->stateDistribution().empiricalMean();
        // 3d models
        for(size_t i = 0; i < object_names_.size(); i++)
        {
            string object_model_path = "package://arm_object_models/objects/" + object_names_[i] + "/" + object_names_[i] + ".obj";
            ri::PublishMarker(mean.homogeneous_matrix(i).cast<float>(),
                              ros_image.header, object_model_path, object_publisher_,
                              i, 1, 0, 0);
        }
    }




private:

    FilterContextPtr filter_context_;

    ros::NodeHandle node_handle_;
    bool is_first_iteration_;
    double previous_time_;

    ros::Publisher object_publisher_;

    vector<vector<size_t> > dependencies_;

    boost::mutex mutex_;

    //	filter::StateFilter standard_filter_;
    boost::shared_ptr<filter::CoordinateFilter> cpu_filter_;

    // parameters
    vector<string> object_names_;
    Matrix3d camera_matrix_;
    int downsampling_factor_;
    int cpu_sample_count_;
    double p_visible_init_;
    double p_visible_visible_;
    double p_visible_occluded_;
};
