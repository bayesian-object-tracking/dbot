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
#include <state_filtering/utils/image_visualizer.hpp>

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
#include <state_filtering/filters/stochastic/coordinate_filter.hpp>
//#include <state_filtering/filters/stochastic/particle_filter_context.hpp>

// observation model
#include <state_filtering/models/measurement/implementations/kinect_measurement_model.hpp>
#include <state_filtering/models/measurement/features/rao_blackwell_measurement_model.hpp>
#include <state_filtering/models/measurement/implementations/image_measurement_model_cpu.hpp>
//#include <state_filtering/models/measurement/implementations/image_measurement_model_gpu/image_measurement_model_gpu.hpp>

// tools
#include <state_filtering/utils/object_file_reader.hpp>
#include <state_filtering/utils/kinematics_from_urdf.hpp>
#include <state_filtering/utils/part_mesh_model.hpp>
#include <state_filtering/utils/helper_functions.hpp>
#include <state_filtering/utils/pcl_interface.hpp>
#include <state_filtering/utils/ros_interface.hpp>
#include <state_filtering/utils/macros.hpp>
//#include "cloud_visualizer.hpp"

// distributions
#include <state_filtering/distributions/distribution.hpp>
#include <state_filtering/distributions/implementations/gaussian.hpp>
#include <state_filtering/models/process/features/stationary_process.hpp>
//#include <state_filtering/models/process/implementations/composed_stationary_process_model.hpp>
#include <state_filtering/models/process/implementations/brownian_object_motion.hpp>

#include <state_filtering/states/rigid_body_system.hpp>
#include <state_filtering/states/floating_body_system.hpp>
#include <state_filtering/states/robot_state.hpp>


using namespace boost;
using namespace std;
using namespace Eigen;
using namespace distributions;





class RobotTracker
{
public:
    typedef double                                                        ScalarType;
    typedef RobotState<>                                                  StateType;

    typedef typename distributions::DampedWienerProcess<ScalarType, Eigen::Dynamic> ProcessType;
    typedef typename distributions::ImageMeasurementModelCPU                        ObserverType;

    typedef typename ProcessType::InputType                                 InputType;
    typedef ObserverType::MeasurementType                                   MeasurementType;
    typedef ObserverType::IndexType                                         IndexType;

    typedef distributions::RaoBlackwellCoordinateParticleFilter
    <ScalarType, StateType, ProcessType, ObserverType> FilterType;

    RobotTracker():
        node_handle_("~"),
        tf_prefix_("MEAN"),
        last_measurement_time_(std::numeric_limits<ScalarType>::quiet_NaN())
    {
        ri::ReadParameter("downsampling_factor", downsampling_factor_, node_handle_);
        ri::ReadParameter("evaluation_count", evaluation_count_, node_handle_);
	
        pub_point_cloud_ = boost::shared_ptr<ros::Publisher>(new ros::Publisher());
        *pub_point_cloud_ = node_handle_.advertise<sensor_msgs::PointCloud2> ("/XTION/depth/points", 5);
    }

    void Initialize(vector<VectorXd> initial_samples_eigen,
                    const sensor_msgs::Image& ros_image,
                    Matrix3d camera_matrix,
                    boost::shared_ptr<KinematicsFromURDF> &urdf_kinematics)
    {
        boost::mutex::scoped_lock lock(mutex_);

        // convert initial samples to our state format
        vector<StateType> initial_samples(initial_samples_eigen.size());
        for(size_t i = 0; i < initial_samples.size(); i++)
            initial_samples[i] = initial_samples_eigen[i];

        // convert camera matrix and image to desired format
        camera_matrix.topLeftCorner(2,3) /= double(downsampling_factor_);
        camera_matrix_ = camera_matrix;
        // TODO: Fix with non-fake arm_rgbd node
        MeasurementType image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_)/ 1000.; // convert to meters

        // read some parameters ---------------------------------------------------------------------------------------------------------
        bool use_gpu; ri::ReadParameter("use_gpu", use_gpu, node_handle_);
        int max_sample_count; ri::ReadParameter("max_sample_count", max_sample_count, node_handle_);
        double p_visible_init; ri::ReadParameter("p_visible_init", p_visible_init, node_handle_);
        double p_visible_visible; ri::ReadParameter("p_visible_visible", p_visible_visible, node_handle_);
        double p_visible_occluded; ri::ReadParameter("p_visible_occluded", p_visible_occluded, node_handle_);
        double joint_angle_sigma; ri::ReadParameter("joint_angle_sigma", joint_angle_sigma, node_handle_);
        double damping; ri::ReadParameter("damping", damping, node_handle_);
        double tail_weight; ri::ReadParameter("tail_weight", tail_weight, node_handle_);
        double model_sigma; ri::ReadParameter("model_sigma", model_sigma, node_handle_);
        double sigma_factor; ri::ReadParameter("sigma_factor", sigma_factor, node_handle_);
        vector<vector<size_t> > sampling_blocks;
        ri::ReadParameter("sampling_blocks", sampling_blocks, node_handle_);
        vector<double> joint_sigmas;
        node_handle_.getParam("joint_sigmas", joint_sigmas);
        double max_kl_divergence; ri::ReadParameter("max_kl_divergence", max_kl_divergence, node_handle_);


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
        dimension_ = robot_state->state_size();

        // initialize the result container for the emperical mean
        mean_ = boost::shared_ptr<RobotState<> > (new RobotState<>(part_meshes_.size(),
                                                  urdf_kinematics->num_joints(),
                                                  urdf_kinematics));

        robot_renderer_ = boost::shared_ptr<obj_mod::RigidBodyRenderer>(
                    new obj_mod::RigidBodyRenderer(part_vertices,
                                                   part_triangle_indices,
                                                   robot_state));

        // FOR DEBUGGING
        std::cout << "Image rows and cols " << image.rows() << " " << image.cols() << std::endl;

        robot_renderer_->state(initial_samples[0]);
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

        boost::shared_ptr<ObserverType> measurement_model;

        // cpu obseration model -----------------------------------------------------------------------------------------------------
        boost::shared_ptr<distributions::KinectMeasurementModel> kinect_measurement_model(
                    new distributions::KinectMeasurementModel(tail_weight, model_sigma, sigma_factor));
        boost::shared_ptr<proc_mod::OcclusionProcess> occlusion_process_model(
                    new proc_mod::OcclusionProcess(1. - p_visible_visible, 1. - p_visible_occluded));
        measurement_model = boost::shared_ptr<ObserverType>(
                    new ObserverType(camera_matrix,
                                             image.rows(),
                                             image.cols(),
                                             initial_samples.size(),
                                             robot_renderer_,
                                             kinect_measurement_model,
                                             occlusion_process_model,
                                             p_visible_init));


        // initialize process model =====================================================================================================
        if(dimension_ != joint_sigmas.size())
        {
            cout << "the dimension of the joint sigmas is " << joint_sigmas.size()
                 << " while the state dimension is " << dimension_ << endl;
            exit(-1);
        }
        boost::shared_ptr<ProcessType> process(new ProcessType(dimension_));
        MatrixXd joint_covariance = MatrixXd::Zero(dimension_, dimension_);
        for(size_t i = 0; i < dimension_; i++)
            joint_covariance(i, i) = pow(joint_sigmas[i], 2);
        process->Parameters(damping, joint_covariance);

        // initialize coordinate_filter =================================================================================================
        filter_ = boost::shared_ptr<FilterType>(
                    new FilterType(process, measurement_model, sampling_blocks, max_kl_divergence));

        // we evaluate the initial particles and resample -------------------------------------------------------------------------------
        cout << "evaluating initial particles cpu ..." << endl;
        filter_->Samples(initial_samples);
        filter_->Filter(image, 0.0, InputType::Zero(dimension_));
        filter_->Resample(evaluation_count_/sampling_blocks.size());
    }

    void Filter(const sensor_msgs::Image& ros_image)
    {
        std::cout << "Calling the filter function " << std::endl;
        boost::mutex::scoped_lock lock(mutex_);

        if(std::isnan(last_measurement_time_))
            last_measurement_time_ = ros_image.header.stamp.toSec();
        ScalarType delta_time = ros_image.header.stamp.toSec() - last_measurement_time_;

        // TODO: THIS IS JUST FOR DEBUGGING, SINCE OTHERWISE LARGE DELAYS
        // MAKE IT GO WILD
        delta_time = 0.03;

        // convert image
        MeasurementType image = ri::Ros2Eigen<ScalarType>(ros_image, downsampling_factor_) / 1000.; // convert to m

        // filter
        INIT_PROFILING;
        filter_->Filter(image, delta_time, VectorXd::Zero(dimension_));
        MEASURE("-----------------> total time for filtering");


	// get the mean estimation for the robot joints
	// Casting is a disgusting hack to make sure that the correct equal-operator is used
	// TODO: Make this right 
	*mean_ = (Eigen::VectorXd)(filter_->StateDistribution().EmpiricalMean());

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

  void publishPointCloud(const MeasurementType&       image,
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

  ScalarType last_measurement_time_;
  

  boost::mutex mutex_;
  ros::NodeHandle node_handle_;

  boost::shared_ptr<FilterType> filter_;
  
  boost::shared_ptr<RobotState<> > mean_;
  boost::shared_ptr<robot_state_pub::RobotStatePublisher> robot_state_publisher_;
  boost::shared_ptr<ros::Publisher> pub_point_cloud_;

  std::string tf_prefix_;
  std::string root_;

  // Camera parameters
  Matrix3d camera_matrix_;
  
  // parameters
  int downsampling_factor_;
  int evaluation_count_;

  int dimension_;

  // For debugging
  boost::shared_ptr<obj_mod::RigidBodyRenderer> robot_renderer_;
};

#endif

