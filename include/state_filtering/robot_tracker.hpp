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

#ifndef ROBOT_TRACKER_
#define ROBOT_TRACKER_


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

// for visualizing the estimated robot state
#include <robot_state_pub/robot_state_publisher.h>
#include <tf/transform_broadcaster.h>


// filter
#include <state_filtering/filter/particle/coordinate_filter.hpp>
#include <state_filtering/filter/particle/particle_filter_context.hpp>

// observation model
#include <state_filtering/observation_models/cpu_image_observation_model/kinect_measurement_model.hpp>
#include <state_filtering/observation_models/image_observation_model.hpp>
#include <state_filtering/observation_models/cpu_image_observation_model/cpu_image_observation_model.hpp>
//#include <state_filtering/observation_models/gpu_image_observation_model/gpu_image_observation_model.hpp>

// tools
#include <state_filtering/tools/object_file_reader.hpp>
#include <state_filtering/tools/kinematics_from_urdf.hpp>
#include <state_filtering/tools/part_mesh_model.hpp>
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
#include <state_filtering/system_states/robot_state.hpp>


using namespace boost;
using namespace std;
using namespace Eigen;
using namespace filter;





class RobotTracker
{
public:
    typedef Eigen::Matrix<double, -1, -1> Image;
    typedef filter::CoordinateFilter FilterType;

    RobotTracker():
        node_handle_("~"),
        is_first_iteration_(true),
        duration_(0),
	tf_prefix_("MEAN")
    {
        //ri::ReadParameter("object_names", object_names_, node_handle_);
        ri::ReadParameter("downsampling_factor", downsampling_factor_, node_handle_);
        ri::ReadParameter("sample_count", sample_count_, node_handle_);
	
	pub_point_cloud_ = boost::shared_ptr<ros::Publisher>(new ros::Publisher());
	*pub_point_cloud_ = node_handle_.advertise<sensor_msgs::PointCloud2> ("/XTION/depth/points", 5);
    }

    void Initialize(vector<VectorXd> single_body_samples,
                    const sensor_msgs::Image& ros_image,
                    Matrix3d camera_matrix,
                    boost::shared_ptr<KinematicsFromURDF> &urdf_kinematics)
    {
        boost::mutex::scoped_lock lock(mutex_);

        // convert camera matrix and image to desired format ----------------------------------------------------------------------------
        camera_matrix.topLeftCorner(2,3) /= double(downsampling_factor_);
	camera_matrix_ = camera_matrix;
        // TODO: Fix with non-fake arm_rgbd node
        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_)/ 1000.; // convert to meters

        // read some parameters ---------------------------------------------------------------------------------------------------------
        bool use_gpu; ri::ReadParameter("use_gpu", use_gpu, node_handle_);
        int max_sample_count; ri::ReadParameter("max_sample_count", max_sample_count, node_handle_);
        double p_visible_init; ri::ReadParameter("p_visible_init", p_visible_init, node_handle_);
        double p_visible_visible; ri::ReadParameter("p_visible_visible", p_visible_visible, node_handle_);
        double p_visible_occluded; ri::ReadParameter("p_visible_occluded", p_visible_occluded, node_handle_);
        double joint_angle_sigma; ri::ReadParameter("joint_angle_sigma", joint_angle_sigma, node_handle_);
        double linear_acceleration_sigma; ri::ReadParameter("linear_acceleration_sigma", linear_acceleration_sigma, node_handle_);
        double angular_acceleration_sigma; ri::ReadParameter("angular_acceleration_sigma", angular_acceleration_sigma, node_handle_);
        double damping; ri::ReadParameter("damping", damping, node_handle_);
        double tail_weight; ri::ReadParameter("tail_weight", tail_weight, node_handle_);
        double model_sigma; ri::ReadParameter("model_sigma", model_sigma, node_handle_);
        double sigma_factor; ri::ReadParameter("sigma_factor", sigma_factor, node_handle_);

        // initialize observation model =================================================================================================

        // Read the URDF for the specific robot and get part meshes
        std::vector<boost::shared_ptr<PartMeshModel> > part_meshes_;
        urdf_kinematics->GetPartMeshes(part_meshes_);
        ROS_INFO("Number of part meshes %d", (int)part_meshes_.size());
        ROS_INFO("Number of joints %d", urdf_kinematics->num_joints());

	// get the name of the root frame
	root_ = urdf_kinematics->GetRootFrameID();

	// initialize the robot state publisher
	robot_state_publisher_ = boost::shared_ptr<robot_state_pub::RobotStatePublisher>
	  (new robot_state_pub::RobotStatePublisher(urdf_kinematics->GetTree()));

        vector<vector<size_t> > dependencies;
        urdf_kinematics->GetDependencies(dependencies);

        vector<vector<Vector3d> > part_vertices(part_meshes_.size());
        vector<vector<vector<int> > > part_triangle_indices(part_meshes_.size());
        for(size_t i = 0; i < part_meshes_.size(); i++)
        {
            std::cout << "The single part added : " << part_meshes_[i]->get_name() << std::endl;
            part_vertices[i] = *(part_meshes_[i]->get_vertices());
            part_triangle_indices[i] = *(part_meshes_[i]->get_indices());
        }


        // the rigid_body_system is essentially the state vector with some convenience functions for retrieving
        // the poses of the rigid objects
        boost::shared_ptr<RigidBodySystem<> > robot_state(new RobotState<>(part_meshes_.size(),
                                                                           urdf_kinematics->num_joints(),
                                                                           urdf_kinematics));

	// initialize the result container for the emperical mean
	mean_ = boost::shared_ptr<RobotState<> > (new RobotState<>(part_meshes_.size(),
								   urdf_kinematics->num_joints(),
								   urdf_kinematics));

        robot_renderer_ = boost::shared_ptr<obj_mod::RigidBodyRenderer>(new obj_mod::RigidBodyRenderer(part_vertices,
												       part_triangle_indices,
												       robot_state));

        // FOR DEBUGGING
	
        std::cout << "Image rows and cols " << image.rows() << " " << image.cols() << std::endl;

        robot_renderer_->state(single_body_samples[0]);
        std::vector<int> indices;
        std::vector<float> depth;
        robot_renderer_->Render(camera_matrix,
				image.rows(),
				image.cols(),
				indices,
				depth);
        vis::ImageVisualizer image_viz(image.rows(),image.cols());
        image_viz.set_image(image);
        image_viz.add_points(indices, depth);
	image_viz.show_image("enchilada ");

	/*
        std::vector<std::vector<Eigen::Vector3d> > vertices = robot_renderer_->vertices();
        vis::CloudVisualizer cloud_vis;
        std::vector<std::vector<Eigen::Vector3d> >::iterator it = vertices.begin();
        for(; it!=vertices.end();++it){
            if(!it->empty())
                cloud_vis.add_cloud(*it);
        }

        cloud_vis.show();
	*/


        boost::shared_ptr<obs_mod::ImageObservationModel> observation_model;
        if(!use_gpu)
        {
            // cpu obseration model -----------------------------------------------------------------------------------------------------
            boost::shared_ptr<obs_mod::KinectMeasurementModel>
                    kinect_measurement_model(new obs_mod::KinectMeasurementModel(tail_weight, model_sigma, sigma_factor));
            boost::shared_ptr<proc_mod::OcclusionProcessModel>
                    occlusion_process_model(new proc_mod::OcclusionProcessModel(1. - p_visible_visible, 1. - p_visible_occluded));
            observation_model = boost::shared_ptr<obs_mod::ImageObservationModel>(new obs_mod::CPUImageObservationModel(
                                                                                      camera_matrix,
                                                                                      image.rows(),
                                                                                      image.cols(),
                                                                                      single_body_samples.size(),
                                                                                      robot_state,
                                                                                      robot_renderer_,
                                                                                      kinect_measurement_model,
                                                                                      occlusion_process_model,
                                                                                      p_visible_init));
        }
        else
        {
            /*
            // gpu obseration model -----------------------------------------------------------------------------------------------------
            boost::shared_ptr<obs_mod::GPUImageObservationModel> gpu_observation_model(new obs_mod::GPUImageObservationModel(
                                                                                           camera_matrix,
                                                                                           image.rows(),
                                                                                           image.cols(),
                                                                                           max_sample_count,
                                                                                           p_visible_init,
                                                                                           robot_state));

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
      */
        }

        // initialize process model =====================================================================================================
        boost::shared_ptr<StationaryProcess<> > process_model;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// hopefully, by just commenting the above and uncommenting the stuff below we should have a process model for the robot joints
        boost::shared_ptr<DampedBrownianMotion<> > joint_process_model(new DampedBrownianMotion<>(robot_state->state_size()));
        MatrixXd joint_covariance = MatrixXd::Identity(joint_process_model->variable_size(),
                                                       joint_process_model->variable_size())
                * pow(joint_angle_sigma, 2);
        joint_process_model->parameters(0., joint_covariance);
        process_model = joint_process_model;
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // initialize coordinate_filter =================================================================================================
        filter_ = boost::shared_ptr<filter::CoordinateFilter>
                (new filter::CoordinateFilter(observation_model, process_model, dependencies));

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // we evaluate the initial particles and resample -------------------------------------------------------------------------------
        cout << "evaluating initial particles cpu ..." << endl;
        filter_->set_states(single_body_samples);
        filter_->Evaluate(image);
        filter_->Resample(sample_count_);
    }

    void Filter(const sensor_msgs::Image& ros_image)
    {
        std::cout << "Calling the filter function " << std::endl;
        boost::mutex::scoped_lock lock(mutex_);

        // convert imagex
        Image image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_) / 1000.; // convert to m

        // the time since start is computed
        if(is_first_iteration_)
        {
            previous_time_ = ros_image.header.stamp.toSec();
            is_first_iteration_ = false;
        }
        duration_ += ros_image.header.stamp.toSec() - previous_time_;

        // filter
        INIT_PROFILING;
        filter_->Enchilada(FilterType::Control::Zero(filter_->control_size()),
                           duration_,
                           image,
                           sample_count_);
        MEASURE("-----------------> total time for filtering");

        previous_time_ = ros_image.header.stamp.toSec();


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// the visualization of the estimated joint angles
        /// and of the point cloud from the depth image

	// get the mean estimation for the robot joints
	*mean_ = filter_->stateDistribution().empiricalMean();

	// DEBUG to see depth images
	robot_renderer_->state(*mean_);
        std::vector<int> indices;
        std::vector<float> depth;
        robot_renderer_->Render(camera_matrix_,
				image.rows(),
				image.cols(),
				indices,
				depth);
	//image_viz_ = boost::shared_ptr<vis::ImageVisualizer>(new vis::ImageVisualizer(image.rows(),image.cols()));
        vis::ImageVisualizer image_viz(image.rows(),image.cols());
        image_viz.set_image(image);
        image_viz.add_points(indices, depth);
	image_viz.show_image("enchilada ", 500, 500, 1.0);
	//////

	std::map<std::string, double> joint_positions;
	mean_->GetJointState(joint_positions);
	ros::Time t = ros::Time::now();
	// publish movable joints
	robot_state_publisher_->publishTransforms(joint_positions,  t, tf_prefix_);
	// make sure there is a valid transformation between base of real robot and estimated robot
	publishTransform(t, root_, tf::resolve(tf_prefix_, root_));
	// publish fixed transforms
	robot_state_publisher_->publishFixedTransforms(tf_prefix_);
	// publish point cloud
	publishPointCloud(image, ros_image.header.frame_id, ros_image.header.stamp);
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }




private:  

  void publishTransform(const ros::Time& time,
			const std::string& from,
			const std::string& to)
  {
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setIdentity();
    br.sendTransform(tf::StampedTransform(transform, time, from, to));
  }

  void publishPointCloud(const Image&       image,
			 const std::string& frame_id,
			 const ros::Time&   stamp)
  {
    
    float bad_point = std::numeric_limits<float>::quiet_NaN();
 
    sensor_msgs::PointCloud2Ptr points = boost::make_shared<sensor_msgs::PointCloud2 > ();
    points->header.frame_id = frame_id;
    points->header.stamp = stamp;
    points->width        = image.cols();
    points->height       = image.rows();
    points->is_dense     = false;
    points->is_bigendian = false;
    points->fields.resize( 3 );
    points->fields[0].name = "x"; 
    points->fields[1].name = "y"; 
    points->fields[2].name = "z";
    int offset = 0;
    for (size_t d = 0; 
	 d < points->fields.size (); 
	 ++d, offset += sizeof(float)) {
      points->fields[d].offset = offset;
      points->fields[d].datatype = 
	sensor_msgs::PointField::FLOAT32;
      points->fields[d].count  = 1;
    }
    
    points->point_step = offset;
    points->row_step   = 
      points->point_step * points->width;
    
    points->data.resize (points->width * 
			 points->height * 
			 points->point_step);
    
    for (size_t u = 0, nRows = image.rows(), nCols = image.cols(); u < nCols; ++u)
      for (size_t v = 0; v < nRows; ++v)
	{
	  //float depth = depth_row[u]/1000.0;
	  float depth = image(v,u);
	  if(depth!=depth)
	    {
	      // depth is invalid
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[0].offset], &bad_point, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[1].offset], &bad_point, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[2].offset], &bad_point, sizeof (float));
	    } 
	  else 
	    {
	      // depth is valid
	      // BAAAD Jeannette BAAAAD. Hard-coded camera parameters TODO: Get this from a camera model
	      float x = ((float)u - 40.0) * depth / 75.0;
	      float y = ((float)v - 30.0) * depth / 75.0;
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[0].offset], &x, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[1].offset], &y, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[2].offset], &depth, sizeof (float));
	    }
	}

    if (  pub_point_cloud_->getNumSubscribers () > 0)
      pub_point_cloud_->publish (points);
  }
  
  double duration_;

  boost::mutex mutex_;
  ros::NodeHandle node_handle_;

  boost::shared_ptr<FilterType> filter_;
  
  boost::shared_ptr<RobotState<> > mean_;
  boost::shared_ptr<robot_state_pub::RobotStatePublisher> robot_state_publisher_;
  boost::shared_ptr<ros::Publisher> pub_point_cloud_;

  bool is_first_iteration_;
  double previous_time_;

  std::string tf_prefix_;
  std::string root_;

  // Camera parameters
  Matrix3d camera_matrix_;
  
  // parameters
  int downsampling_factor_;
  int sample_count_;

  // For debugging
  boost::shared_ptr<obj_mod::RigidBodyRenderer> robot_renderer_;
};

#endif

