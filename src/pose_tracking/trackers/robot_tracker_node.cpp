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


//#define PROFILING_ON

#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

#include <fast_filtering/state_filtering.hpp>

#include <pose_tracking/trackers/robot_tracker.hpp>
#include <pose_tracking/utils/cloud_visualizer.hpp>
#include <pose_tracking/utils/kinematics_from_urdf.hpp>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>

typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;

class RobotTrackerNode
{
  ros::NodeHandle nh_;
  ros::NodeHandle priv_nh_;

  RobotTracker robot_tracker_;
  
  std::string depth_image_topic_;
  std::string camera_info_topic_;
  int initial_sample_count_;

  sensor_msgs::Image ros_image_;

  sensor_msgs::JointState joint_state_;
  sensor_msgs::JointState joint_state_copy_;
  boost::mutex joint_state_mutex_;
  
  ros::Subscriber subscriber_;

  bool first_time_;
  bool has_image_;
  bool has_joints_;

public:
  RobotTrackerNode()
    : priv_nh_("~")
    , first_time_(true)
    , has_image_(false)
    , has_joints_(false)
  {
    // subscribe to the joint angles
    ros::Subscriber joint_states_sub = nh_.subscribe<sensor_msgs::JointState>("/joint_states", 
									      1,
									      &RobotTrackerNode::jointStateCallback, 
									      this);
    // initialize the kinematics 
    boost::shared_ptr<KinematicsFromURDF> urdf_kinematics(new KinematicsFromURDF());

    // read the node parameters
    ri::ReadParameter("depth_image_topic", depth_image_topic_, priv_nh_);
    ri::ReadParameter("camera_info_topic", camera_info_topic_, priv_nh_);
    ri::ReadParameter("initial_sample_count", initial_sample_count_, priv_nh_);
    
    ros::Subscriber depth_image_sub = nh_.subscribe<sensor_msgs::Image>(depth_image_topic_, 
									     1,
									     &RobotTrackerNode::depthImageCallback, 
									     this);
    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Zero();
    // get the camera parameters
    while(camera_matrix.sum() == 0.0)
      camera_matrix = ri::GetCameraMatrix<double>(camera_info_topic_, nh_, 2.0);
    
    while(!(has_joints_ & has_image_))
      {
	ROS_INFO("Waiting for joint angles and depth images: %d %d", has_joints_, has_image_);
	ros::spinOnce();
	usleep(10000);
      }

    std::vector<Eigen::VectorXd> initial_states;
    if(initial_sample_count_>1)
      initial_states = urdf_kinematics->GetInitialSamples(joint_state_copy_, initial_sample_count_);
    else
      initial_states = urdf_kinematics->GetInitialJoints(joint_state_copy_);

    // intialize the filter
    robot_tracker_.Initialize(initial_states, ros_image_, camera_matrix, urdf_kinematics);
    std::cout << "done initializing" << std::endl;
   
    
    subscriber_ = nh_.subscribe(depth_image_topic_, 
			       1, 
			       &RobotTracker::Filter, 
			       &robot_tracker_);
    
  }
  
  void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg)
  {
    ros_image_ = *msg;
    if(!has_image_)
      has_image_=true;
    {
      // get the latest corresponding joint angles
      boost::mutex::scoped_lock lock(joint_state_mutex_);
      if(!first_time_)
	{
	  joint_state_copy_ = joint_state_;
	  has_joints_=true;
	}
    }
  }
  
  void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    boost::mutex::scoped_lock lock(joint_state_mutex_);
    joint_state_ = *msg;
    
    if (first_time_)
      first_time_ = false;
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robot_tracker");
  RobotTrackerNode rt;
  ros::spin();
  return 0;
}
