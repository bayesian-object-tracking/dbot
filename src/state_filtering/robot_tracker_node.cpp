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

#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

#include <state_filtering/robot_tracker.hpp>
#include <state_filtering/tools/cloud_visualizer.hpp>
#include <state_filtering/tools/kinematics_from_urdf.hpp>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>



typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;
typedef Eigen::Matrix<double, -1, -1> Image;

class RobotTrackerNode
{
  ros::NodeHandle nh_;
  
  string depth_image_topic_;
  string camera_info_topic_;
  int initial_sample_count_;

  Matrix3d camera_matrix_;

  sensor_msgs::JointState joint_state_;
  boost::mutex joint_state_mutex_;

public:
  RobotTrackerNode()
    : nh_("~")
  {
    // subscribe to the joint angles
    ros::Subscriber joint_states_sub = nh_.subscribe<sensor_msgs::JointState>("/joint_states", 
									      1,
									      &RobotTrackerNode::jointStateCallback, 
									      this);
    // initialize the kinematics 
    boost::shared_ptr<KinematicsFromURDF> urdf_kinematics(new KinematicsFromURDF());
   
    // read the node parameters
    ri::ReadParameter("depth_image_topic", depth_image_topic_, nh_);
    ri::ReadParameter("camera_info_topic", camera_info_topic_, nh_);
    ri::ReadParameter("initial_sample_count", initial_sample_count_, nh_);
    
    // get the camera parameters
    camera_matrix_ = ri::GetCameraMatrix<double>(camera_info_topic_, nh_, 2.0);

    // subscribe to the image topic
    sensor_msgs::Image::ConstPtr ros_image(new sensor_msgs::Image);
    /*
    // get observations from camera
    sensor_msgs::Image::ConstPtr ros_image =
    ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic, node_handle, ros::Duration(10.0));
    
    // get the latest corresponding joint angles
    {
    boost::mutex::scoped_lock lock(joint_state_mutex_);
    std::cout << joint_state_ << std::endl;
    }
    
    Image image = ri::Ros2Eigen<double>(*ros_image) / 1000.; // convert to m
    */

    vector<VectorXd> initial_states;
    /// ====================================================================================================
    /// TODO: this has to be adapted, we have to provide some samples around the initial robot joint angles
    /*vector<VectorXd> initial_states = pi::SampleTableClusters(hf::Image2Points(image, camera_matrix),
      initial_sample_count);
    */
    /// ====================================================================================================
    


    // intialize the filter
    RobotTracker robot_tracker;
    robot_tracker.Initialize(initial_states, *ros_image, camera_matrix_, urdf_kinematics);
    cout << "done initializing" << endl;

    ros::Subscriber subscriber = nh_.subscribe(depth_image_topic_, 1, &RobotTracker::Filter, &robot_tracker);
  }
  
  void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    boost::mutex::scoped_lock lock(joint_state_mutex_);
    joint_state_ = *msg;
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robot_tracker");
  RobotTrackerNode rt;
  ros::spin();
  return 0;
}
