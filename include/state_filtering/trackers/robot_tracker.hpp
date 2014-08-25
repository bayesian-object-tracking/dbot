/*************************************************************************
This software allows for filtering in high-dimensional observation and
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
#include <image_transport/image_transport.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl-1.6/pcl/ros/conversions.h>
#include <pcl-1.6/pcl/point_cloud.h>
#include <pcl-1.6/pcl/point_types.h>

// for visualizing the estimated robot state
#include <robot_state_pub/robot_state_publisher.h>
#include <tf/transform_broadcaster.h>


// filter
#include <state_filtering/filters/stochastic/rao_blackwell_coordinate_particle_filter.hpp>
//#include <state_filtering/filters/stochastic/particle_filter_context.hpp>

// observation model
#include <state_filtering/models/observers/implementations/kinect_observer.hpp>
#include <state_filtering/models/observers/features/rao_blackwell_observer.hpp>
#include <state_filtering/models/observers/implementations/image_observer_cpu.hpp>
//#include <state_filtering/models/observers/implementations/image_observer_gpu/image_observer_gpu.hpp>

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

#include <state_filtering/distributions/implementations/gaussian.hpp>
#include <state_filtering/models/processes/features/stationary_process.hpp>
//#include <state_filtering/models/processes/implementations/composed_stationary_process_model.hpp>
#include <state_filtering/models/processes/implementations/brownian_object_motion.hpp>

#include <state_filtering/states/rigid_body_system.hpp>
#include <state_filtering/states/floating_body_system.hpp>
#include <state_filtering/states/robot_state.hpp>


using namespace boost;
using namespace std;
using namespace Eigen;
using namespace sf;

class RobotTracker
{
public:
    typedef double          Scalar;
    typedef RobotState<>    StateType;

    typedef sf::DampedWienerProcess<StateType>      ProcessModel;
    typedef sf::ImageObserverCPU<Scalar, StateType> ObservationModel;

    typedef typename ProcessModel::InputVector      InputVector;
    typedef typename ObservationModel::Observation  Observation;

    typedef sf::RaoBlackwellCoordinateParticleFilter
                <ProcessModel, ObservationModel> FilterType;

    RobotTracker():
        node_handle_("~"),
        tf_prefix_("MEAN"),
        last_measurement_time_(std::numeric_limits<Scalar>::quiet_NaN())
    {
        ri::ReadParameter("downsampling_factor", downsampling_factor_, node_handle_);
        ri::ReadParameter("evaluation_count", evaluation_count_, node_handle_);
        ri::ReadParameter("camera_frame", camera_frame_, node_handle_);
	
	pub_point_cloud_ = boost::shared_ptr<ros::Publisher>(new ros::Publisher());
	*pub_point_cloud_ = node_handle_.advertise<sensor_msgs::PointCloud2> ("/XTION/depth/points", 5);
	
	
	boost::shared_ptr<image_transport::ImageTransport> it(new image_transport::ImageTransport(node_handle_));
	pub_rgb_image_ = it->advertise ("/XTION/depth/image_color", 5);
	
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
        Observation image = ri::Ros2Eigen<double>(ros_image, downsampling_factor_)/ 1000.; // convert to meters

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

        std::vector<std::string> joints = urdf_kinematics->GetJointMap();
        hf::PrintVector(joints);

	
        // get the name of the root frame
        root_ = urdf_kinematics->GetRootFrameID();

        // initialize the robot state publisher
        robot_state_publisher_ = boost::shared_ptr<robot_state_pub::RobotStatePublisher>
        (new robot_state_pub::RobotStatePublisher(urdf_kinematics->GetTree()));

        vector<vector<Vector3d> > part_vertices(part_meshes_.size());
        vector<vector<vector<int> > > part_triangle_indices(part_meshes_.size());
        for(size_t i = 0; i < part_meshes_.size(); i++)
        {
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

        boost::shared_ptr<ObservationModel> observation_model;

        // cpu obseration model -----------------------------------------------------------------------------------------------------
        boost::shared_ptr<sf::KinectObserver> kinect_observer(
                    new sf::KinectObserver(tail_weight, model_sigma, sigma_factor));
        boost::shared_ptr<sf::OcclusionProcess> occlusion_process_model(
                    new sf::OcclusionProcess(1. - p_visible_visible, 1. - p_visible_occluded));
        observation_model = boost::shared_ptr<ObservationModel>(
                    new ObservationModel(camera_matrix,
                                             image.rows(),
                                             image.cols(),
                                             initial_samples.size(),
                                             robot_renderer_,
                                             kinect_observer,
                                             occlusion_process_model,
                                             p_visible_init));


        // initialize process model =====================================================================================================
        if(dimension_ != joint_sigmas.size())
        {
            cout << "the dimension of the joint sigmas is " << joint_sigmas.size()
                 << " while the state dimension is " << dimension_ << endl;
            exit(-1);
        }
        boost::shared_ptr<ProcessModel> process(new ProcessModel(dimension_));
        MatrixXd joint_covariance = MatrixXd::Zero(dimension_, dimension_);
        for(size_t i = 0; i < dimension_; i++)
            joint_covariance(i, i) = pow(joint_sigmas[i], 2);
        process->Parameters(damping, joint_covariance);

        // initialize coordinate_filter =================================================================================================
        filter_ = boost::shared_ptr<FilterType>(
                    new FilterType(process, observation_model, sampling_blocks, max_kl_divergence));

        // we evaluate the initial particles and resample -------------------------------------------------------------------------------
        cout << "evaluating initial particles cpu ..." << endl;
        filter_->Samples(initial_samples);
        filter_->Filter(image, 0.0, InputVector::Zero(dimension_));
        filter_->Resample(evaluation_count_/sampling_blocks.size());
    }

    void Filter(const sensor_msgs::Image& ros_image)
    {
        std::cout << "Calling the filter function " << std::endl;
        boost::mutex::scoped_lock lock(mutex_);

        if(std::isnan(last_measurement_time_))
            last_measurement_time_ = ros_image.header.stamp.toSec();
        Scalar delta_time = ros_image.header.stamp.toSec() - last_measurement_time_;

        // TODO: THIS IS JUST FOR DEBUGGING, SINCE OTHERWISE LARGE DELAYS
        // MAKE IT GO WILD
        delta_time = 0.03;

        // convert image
        Observation image = ri::Ros2Eigen<Scalar>(ros_image, downsampling_factor_) / 1000.; // convert to m

        // filter
        INIT_PROFILING;
        filter_->Filter(image, delta_time, VectorXd::Zero(dimension_));
        MEASURE("-----------------> total time for filtering");


	// get the mean estimation for the robot joints
	// Casting is a disgusting hack to make sure that the correct equal-operator is used
	// TODO: Make this right 
    *mean_ = (Eigen::VectorXd)(filter_->StateDistribution().Mean());

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
	//image_viz.show_image("enchilada ", 500, 500, 1.0);
	//////

	std::map<std::string, double> joint_positions;
	mean_->GetJointState(joint_positions);


	ros::Time t = ros::Time::now();
	// publish movable joints
	robot_state_publisher_->publishTransforms(joint_positions,  t, tf_prefix_);
	// make sure there is a identity transformation between base of real robot and estimated robot
	publishTransform(t, root_, tf::resolve(tf_prefix_, root_));
	// publish fixed transforms
	robot_state_publisher_->publishFixedTransforms(tf_prefix_);
	// publish image
	sensor_msgs::Image overlay;
	image_viz.get_image(overlay);
	publishImage(t, overlay);
	
	// publish point cloud
	publishPointCloud(image, t);
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

private:  

  void publishImage(const ros::Time& time,
		    sensor_msgs::Image &image)
  {
    image.header.frame_id = tf::resolve(tf_prefix_, camera_frame_);
    image.header.stamp = time;
    pub_rgb_image_.publish (image);
  }

  void publishTransform(const ros::Time& time,
			const std::string& from,
			const std::string& to)
  {
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setIdentity();
    br.sendTransform(tf::StampedTransform(transform, time, from, to));
  }

  void publishPointCloud(const Observation&       image,
			 const ros::Time&             stamp)
  {
    
    float bad_point = std::numeric_limits<float>::quiet_NaN();
 
    sensor_msgs::PointCloud2Ptr points = boost::make_shared<sensor_msgs::PointCloud2 > ();
    points->header.frame_id =  tf::resolve(tf_prefix_, camera_frame_);
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
	      float x = ((float)u - camera_matrix_(0,2)) * depth / camera_matrix_(0,0);
	      float y = ((float)v - camera_matrix_(1,2)) * depth / camera_matrix_(1,1);
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[0].offset], &x, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[1].offset], &y, sizeof (float));
	      memcpy (&points->data[v * points->row_step + u * points->point_step + points->fields[2].offset], &depth, sizeof (float));
	    }
	}

    if (  pub_point_cloud_->getNumSubscribers () > 0)
      pub_point_cloud_->publish (points);
  }

  Scalar last_measurement_time_;
  

  boost::mutex mutex_;
  ros::NodeHandle node_handle_;

  boost::shared_ptr<FilterType> filter_;
  
  boost::shared_ptr<RobotState<> > mean_;
  boost::shared_ptr<robot_state_pub::RobotStatePublisher> robot_state_publisher_;
  boost::shared_ptr<ros::Publisher> pub_point_cloud_;
  
  image_transport::Publisher pub_rgb_image_;

  std::string tf_prefix_;
  std::string root_;

  // Camera parameters
  Matrix3d camera_matrix_;
  std::string camera_frame_;


  // parameters
  int downsampling_factor_;
  int evaluation_count_;

  int dimension_;

  // For debugging
  boost::shared_ptr<obj_mod::RigidBodyRenderer> robot_renderer_;
};

#endif

