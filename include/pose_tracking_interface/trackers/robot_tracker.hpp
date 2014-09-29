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

#ifndef POSE_TRACKING_INTERFACE_TRACKERS_ROBOT_TRACKER_HPP
#define POSE_TRACKING_INTERFACE_TRACKERS_ROBOT_TRACKER_HPP

#include <boost/thread/mutex.hpp>

#include <ros/ros.h>

#include <Eigen/Dense>

#include <image_transport/image_transport.h>

#include <robot_state_pub/robot_state_publisher.h>

#include <fast_filtering/filters/stochastic_filters/rao_blackwell_coordinate_particle_filter.hpp>
#include <fast_filtering/models/process_models/damped_wiener_process_model.hpp>
#include <pose_tracking/models/observation_models/kinect_image_observation_model_cpu.hpp>
#include <pose_tracking/states/robot_state.hpp>
#include <pose_tracking_interface/utils/kinematics_from_urdf.hpp>
#include <pose_tracking/utils/rigid_body_renderer.hpp>

#ifdef BUILD_GPU
#include <pose_tracking/models/observation_models/kinect_image_observation_model_gpu/kinect_image_observation_model_gpu.hpp>
#include <pose_tracking/models/observation_models/kinect_image_observation_model_gpu/kinect_image_observation_model_gpu_hack.hpp>
#endif


class RobotTracker
{
public:
    typedef RobotState<>    State;
    typedef State::Scalar   Scalar;

    // process model
    typedef ff::DampedWienerProcessModel<State>         ProcessModel;
    typedef typename ProcessModel::Input                Input;

    // observation models
    typedef ff::KinectImageObservationModelCPU<Scalar,
                                                State>  ObservationModelCPUType;
#ifdef BUILD_GPU
    typedef ff::KinectImageObservationModelGPUHack<State>   ObservationModelGPUType;
#endif
    typedef ObservationModelCPUType::Base ObservationModel;
    typedef ObservationModelCPUType::Observation Observation;

    typedef ff::RaoBlackwellCoordinateParticleFilter<ProcessModel, ObservationModel> FilterType;

    RobotTracker();

    void Initialize(std::vector<Eigen::VectorXd> initial_samples_eigen,
                    const sensor_msgs::Image& ros_image,
                    Eigen::Matrix3d camera_matrix,
                    boost::shared_ptr<KinematicsFromURDF> &urdf_kinematics);

    void Filter(const sensor_msgs::Image& ros_image);

private:  

  void publishImage(const ros::Time& time,
            sensor_msgs::Image &image);

  void publishTransform(const ros::Time& time,
			const std::string& from,
            const std::string& to);

  void publishPointCloud(const Observation& image,
                         const ros::Time& stamp);

  Scalar last_measurement_time_;

  boost::shared_ptr<KinematicsFromURDF> urdf_kinematics_;
  

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
  Eigen::Matrix3d camera_matrix_;
  std::string camera_frame_;


  // parameters
  int downsampling_factor_;
  int evaluation_count_;

  int dimension_;

  // determines whether it is necessary to convert depth data to meters
  bool data_in_meters_;

  // For debugging
  boost::shared_ptr<ff::RigidBodyRenderer> robot_renderer_;  
};

#endif

