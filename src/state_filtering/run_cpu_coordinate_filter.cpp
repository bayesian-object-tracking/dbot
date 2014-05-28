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
// observation model
#include <state_filtering/observation_models/cpu_image_observation_model/gaussian_pixel_observation_model.hpp>
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
#include <state_filtering/distribution/gaussian/gaussian_distribution.hpp>
#include <state_filtering/process_model/stationary_process_model.hpp>
#include <state_filtering/process_model/composed_stationary_process_model.hpp>
#include <state_filtering/process_model/brownian_process_model.hpp>

#include <state_filtering/system_states/rigid_body_system.hpp>
#include <state_filtering/system_states/full_rigid_body_system.hpp>

using namespace boost;
using namespace std;
using namespace Eigen;
using namespace filter;

template<typename T>
void ReadParam(const string& path, T& parameter, ros::NodeHandle node_handle)
{
    cout << "reading parameter from " << path << endl;
    XmlRpc::XmlRpcValue ros_parameter;

    node_handle.getParam(path, ros_parameter);
    parameter = T(ros_parameter);
    cout << "parameter is  " << parameter << endl;
}
template<>
void ReadParam< vector<string> >(const string& path, vector<string>& parameter, ros::NodeHandle node_handle)
{
    cout << "reading parameter from " << path << endl;
    XmlRpc::XmlRpcValue ros_parameter;
    node_handle.getParam(path, ros_parameter);
    parameter.resize(ros_parameter.size());
    cout << "parameter is  (";
    for(size_t i = 0; i < parameter.size(); i++)
    {
        parameter[i] = string(ros_parameter[i]);
        cout << parameter[i] << ", ";
    }
    cout << ")" << endl;
}
template<>
void ReadParam< vector<vector<size_t> > >(const string& path, vector<vector<size_t> >& parameter, ros::NodeHandle node_handle)
{
    cout << "reading parameter from " << path << endl;
    XmlRpc::XmlRpcValue ros_parameter;
    node_handle.getParam(path, ros_parameter);
    parameter.resize(ros_parameter.size());
    cout << "parameter is  (";
    for(size_t i = 0; i < parameter.size(); i++)
    {
        cout << "[ ";
        parameter[i].resize(ros_parameter[i].size());
        for(size_t j = 0; j < parameter[i].size(); j++)
        {
            parameter[i][j] = int(ros_parameter[i][j]);
            cout << parameter[i][j] << ", ";
        }
        cout << "] " ;
    }
    cout << ")" << endl;
}



class TestFilter
{
    template<typename T>
    void ReadParameter(const string& path, T& parameter)
    {
        ReadParam<T>(path, parameter, node_handle_);
    }

public:
    TestFilter(Matrix3d camera_matrix):
        node_handle_("~"),
        is_first_iteration_(true)
    {
        // read parameters ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ReadParameter("dependencies", dependencies_);

        ReadParameter("object_names", object_names_);
        ReadParameter("downsampling_factor", downsampling_factor_);
        ReadParameter("cpu_sample_count", cpu_sample_count_);
        ReadParameter("p_visible_init", p_visible_init_);
        ReadParameter("p_visible_visible", p_visible_visible_);
        ReadParameter("p_visible_occluded", p_visible_occluded_);

        camera_matrix_ = camera_matrix;
        camera_matrix_.topLeftCorner(2,3) /= float(downsampling_factor_);
        object_publisher_ = node_handle_.advertise<visualization_msgs::Marker>("object_model", 0);
    }

    void Initialize(
            vector<VectorXd> single_body_samples,
            sensor_msgs::PointCloud2 ros_cloud)
    {
        boost::mutex::scoped_lock lock(mutex_);
        vector<float> observations; size_t n_rows, n_cols;
        pi::Ros2Std(ros_cloud, downsampling_factor_, observations, n_rows, n_cols);

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
        // pixel_observation_model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        double tail_weight; ReadParameter("tail_weight", tail_weight);
        double model_sigma; ReadParameter("model_sigma", model_sigma);
        double sigma_factor; ReadParameter("sigma_factor", sigma_factor);
        boost::shared_ptr<obs_mod::PixelObservationModel>
                pixel_observation_model(new obs_mod::GaussianPixelObservationModel(tail_weight, model_sigma, sigma_factor));

        // initialize occlusion process model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        boost::shared_ptr<proc_mod::OcclusionProcessModel>
                occlusion_process_model(new proc_mod::OcclusionProcessModel(p_visible_visible_, p_visible_occluded_));

        // cpu obseration model -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        boost::shared_ptr<obs_mod::ImageObservationModel> cpu_observation_model(new obs_mod::CPUImageObservationModel(
                                                                                    camera_matrix_,
                                                                                    n_rows,
                                                                                    n_cols,
                                                                                    single_body_samples.size(),
                                                                                    rigid_body_system,
                                                                                    object_renderer,
                                                                                    pixel_observation_model,
                                                                                    occlusion_process_model,
                                                                                    p_visible_init_));

        // initialize process model ========================================================================================================================================================================================================================================================================================================================================================================================================================
        double free_damping; ReadParameter("free_damping", free_damping);

        double free_linear_acceleration_sigma; ReadParameter("free_linear_acceleration_sigma", free_linear_acceleration_sigma);
        MatrixXd free_linear_acceleration_covariance =
                MatrixXd::Identity(3, 3) * pow(double(free_linear_acceleration_sigma), 2);

        double free_angular_acceleration_sigma; ReadParameter("free_angular_acceleration_sigma", free_angular_acceleration_sigma);
        MatrixXd free_angular_acceleration_covariance =
                MatrixXd::Identity(3, 3) * pow(double(free_angular_acceleration_sigma), 2);

        vector<boost::shared_ptr<StationaryProcessModel<> > > partial_process_models(object_names_.size());
        for(size_t i = 0; i < partial_process_models.size(); i++)
        {
            boost::shared_ptr<BrownianProcessModel<> > partial_process_model(new BrownianProcessModel<>);
            partial_process_model->parameters(
                        object_renderer->get_coms(i).cast<double>(),
                        free_damping,
                        free_linear_acceleration_covariance,
                        free_angular_acceleration_covariance);
            partial_process_models[i] = boost::dynamic_pointer_cast<StationaryProcessModel<> >(partial_process_model);
        }

        boost::shared_ptr<ComposedStationaryProcessModel> process_model
                (new ComposedStationaryProcessModel(partial_process_models));


        // initialize coordinate_filter ============================================================================================================================================================================================================================================================
        cpu_filter_ = boost::shared_ptr<filter::CoordinateFilter>
                (new filter::CoordinateFilter(cpu_observation_model, process_model, dependencies_));


        // create the multi body initial samples ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        FullRigidBodySystem<> default_state(object_names_.size());
        for(size_t object_index = 0; object_index < object_names_.size(); object_index++)
            default_state.translation(object_index) = Vector3d(0, 0, 1.5); // outside of image

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
            cpu_filter_->Evaluate(observations);
            cpu_filter_->Resample(multi_body_samples.size());
            cpu_filter_->get(multi_body_samples);
        }

        // we evaluate the initial particles and resample ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cout << "evaluating initial particles cpu ..." << endl;
        cpu_filter_->set_states(multi_body_samples);
        cpu_filter_->Evaluate(observations);
        cpu_filter_->Resample(cpu_sample_count_);
    }


    void Filter(sensor_msgs::PointCloud2 ros_cloud)
    {
        boost::mutex::scoped_lock lock(mutex_);

        // the time since start is computed ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if(is_first_iteration_)
        {
            start_time_ = ros_cloud.header.stamp.toSec();
            is_first_iteration_ = false;
        }
        double time_since_start = ros_cloud.header.stamp.toSec() - start_time_;

        // the point cloud is converted and downsampled ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        vector<float> observations; size_t n_rows, n_cols;
        pi::Ros2Std(ros_cloud, downsampling_factor_, observations, n_rows, n_cols);

        // filter: this is where stuff happens ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        INIT_PROFILING;
        // for propagation the arguments are the control, which is zero here since we are not using the robot, and the time since start
        cpu_filter_->Propagate(VectorXd::Zero(object_names_.size()*6), time_since_start);
        cpu_filter_->Evaluate(observations, time_since_start, true);
        cpu_filter_->Resample(cpu_sample_count_);
        MEASURE("-----------------> total time for filtering");

        // we visualize the likeliest state -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        vector<float> cpu_loglikes; cpu_filter_->get(cpu_loglikes);
        FullRigidBodySystem<> cpu_likeliest_state = cpu_filter_->get_state(hf::BoundIndex(cpu_loglikes, true));
        // 3d models
        for(size_t i = 0; i < object_names_.size(); i++)
        {
            string object_model_path = "package://arm_object_models/objects/" + object_names_[i] + "/" + object_names_[i] + ".obj";
            ri::PublishMarker(cpu_likeliest_state.get_homogeneous_matrix(i).cast<float>(),
                              ros_cloud.header, object_model_path, object_publisher_,
                              i,
                              1, 0, 0);
        }
        // occlusions
        vector<float> cpu_occlusions = cpu_filter_->get_occlusions(hf::BoundIndex(cpu_loglikes, true));
        vis::ImageVisualizer cpu_vis(n_rows, n_cols);
        cpu_vis.set_image(cpu_occlusions);
        cpu_vis.show_image("occlusions 1", 640, 480, 1);
    }



private:
    ros::NodeHandle node_handle_;
    bool is_first_iteration_;
    double start_time_;
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

int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_filter");
    ros::NodeHandle node_handle("~");

    // read params --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    string point_cloud_topic; ReadParam("point_cloud_topic", point_cloud_topic, node_handle);
    string camera_info_topic; ReadParam("camera_info_topic", camera_info_topic, node_handle);
    int initial_sample_count; ReadParam("initial_sample_count", initial_sample_count, node_handle);

    // read camera_matrix ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cout << "reading camera matrix" << endl;
    sensor_msgs::CameraInfo::ConstPtr camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>
            (camera_info_topic, node_handle, ros::Duration(2.0));
    Matrix3d camera_matrix;
    for(unsigned int col = 0; col < 3; col++)
        for(unsigned int row = 0; row < 3; row++)
            camera_matrix(row,col) = camera_info->K[col+row*3];
    cout << camera_matrix << endl;

    // get observations from camera ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cout << "reading point cloud " << endl;
    sensor_msgs::PointCloud2 ros_cloud  =
            *ros::topic::waitForMessage<sensor_msgs::PointCloud2>(point_cloud_topic, node_handle, ros::Duration(2.0));
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg (ros_cloud, *pcl_cloud);
    cout << "done" << endl;
    vector<Vector3d> all_points;
    size_t all_rows, all_cols;
    pi::Pcl2Eigen(*pcl_cloud, all_points, all_rows, all_cols);

    // find points on table and cluster them ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<Vector3d> table_points;
    size_t table_rows, table_cols;
    Vector4d table_plane;
    pi::PointsOnPlane(all_points, all_rows, all_cols, table_points, table_rows, table_cols, table_plane, true);
    if(table_plane.topRows(3).dot(Eigen::Vector3d(0,1,0)) > 0)
        table_plane = - table_plane;
    table_plane /= table_plane.topRows(3).norm();
    Vector3d table_normal = table_plane.topRows(3);
    vector<vector<Vector3d> > clusters;
    pi::Cluster(table_points, table_rows, table_cols, clusters);
    if(clusters.size() == 0)
    {
        cout << "no objects found on table " << endl;
        return 0;
    }

    // we create samples around the clusters on the table --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<VectorXd> initial_states;
    // create gaussian for sampling
    double standard_deviation_translation = 0.03;
    double standard_deviation_rotation = 100.0;
    GaussianDistribution<double, 1> unit_gaussian;
    unit_gaussian.mean(GaussianDistribution<double, 1>::VariableType::Zero());
    unit_gaussian.covariance(GaussianDistribution<double, 1>::CovarianceType::Identity());

    cout << "found " << clusters.size() << " clusters on table, we will sample around each cluster" 	<< endl;
    for(size_t cluster_index = 0; cluster_index < clusters.size(); cluster_index++)
    {
        Vector3d com(0,0,0);
        for(unsigned int i = 0; i < clusters[cluster_index].size(); i++)
            com += clusters[cluster_index][i];
        com /= float(clusters[cluster_index].size());

        Vector3d t_mean = com - (com.dot(table_normal)+table_plane(3))*table_normal; // project center of mass in table plane
        Vector3d table_vector_a = table_normal.cross(Vector3d(1,1,1)).normalized(); // vector along table plane
        Vector3d table_vector_b = table_normal.cross(table_vector_a); // second vector along table plane
        Matrix3d R_mean; R_mean.col(0) = table_vector_a; R_mean.col(1) = table_vector_b; R_mean.col(2) = table_normal;

        // sample around mean
        for(size_t i = 0; i < size_t(initial_sample_count)/clusters.size(); i++)
        {
            FullRigidBodySystem<-1> state(1);
            state.translation() =
                    t_mean +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_a +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_b;
            state.orientation() = Quaterniond(
                        AngleAxisd(standard_deviation_rotation * unit_gaussian.sample()(0), table_normal) * R_mean).coeffs();

            initial_states.push_back(state);
        }
    }

    // intialize the filter ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    TestFilter test_filter(camera_matrix);
    test_filter.Initialize(initial_states, ros_cloud);
    cout << "done initializing" << endl;

    ros::Subscriber subscriber =
            node_handle.subscribe(point_cloud_topic, 1, &TestFilter::Filter, &test_filter);

    ros::spin();
    return 0;
}
