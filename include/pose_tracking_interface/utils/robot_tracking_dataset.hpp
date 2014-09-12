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

#ifndef POSE_TRACKING_INTERFACE_UTILS_ROBOT_TRACKING_DATASET_HPP
#define POSE_TRACKING_INTERFACE_UTILS_ROBOT_TRACKING_DATASET_HPP

#include <pose_tracking_interface/utils/tracking_dataset.hpp>

class RobotTrackingDataset :  public TrackingDataset
{
public:
  
  RobotTrackingDataset(const std::string& path);

  void AddFrame(const sensor_msgs::Image::ConstPtr& image,
		const sensor_msgs::CameraInfo::ConstPtr& info,
		const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
		const sensor_msgs::JointState::ConstPtr& noisy_joints,
		const Eigen::VectorXd& ground_truth = Eigen::VectorXd(),
		const Eigen::VectorXd& deviation = Eigen::VectorXd());

  void AddFrame(const sensor_msgs::Image::ConstPtr& image,
		const sensor_msgs::CameraInfo::ConstPtr& info,
		const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
		const sensor_msgs::JointState::ConstPtr& noisy_joints);

  Eigen::VectorXd GetDeviation(const size_t& index);
  
  sensor_msgs::JointState::ConstPtr GetGroundTruthJoints(const size_t& index);

  sensor_msgs::JointState::ConstPtr GetNoisyJoints(const size_t& index);
  
  void Load();
  
  void Store();
  
private:
  
  const std::string ground_truth_joints_topic_;
  const std::string noisy_joints_topic_;
  const std::string deviation_filename_;
};

#endif
