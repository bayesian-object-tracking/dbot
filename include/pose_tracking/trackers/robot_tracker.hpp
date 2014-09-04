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

#ifndef STATE_FILTERING_ROBOT_TRACKER_
#define STATE_FILTERING_ROBOT_TRACKER_

#include <boost/thread/mutex.hpp>

#include <ros/ros.h>

#include <Eigen/Dense>

#include <image_transport/image_transport.h>

#include <robot_state_pub/robot_state_publisher.h>

#include <state_filtering/filters/stochastic/rao_blackwell_coordinate_particle_filter.hpp>

#include <tracking/states/robot_state.hpp>
#include <tracking/utils/kinematics_from_urdf.hpp>
#include <tracking/utils/rigid_body_renderer.hpp>

class RobotTracker
{
public:
    typedef RobotState<>    State;
    typedef State::Scalar   Scalar;

    typedef sf::DampedWienerProcess<State>      ProcessModel;
    typedef sf::ImageObserverCPU<Scalar, State> ObservationModel;

    typedef typename ProcessModel::Input            Input;
    typedef typename ObservationModel::Observation  Observation;

    typedef sf::RaoBlackwellCoordinateParticleFilter<ProcessModel, ObservationModel> FilterType;

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

  // For debugging
  boost::shared_ptr<sf::RigidBodyRenderer> robot_renderer_;
};

#endif

