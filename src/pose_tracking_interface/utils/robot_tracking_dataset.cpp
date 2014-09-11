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

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <boost/foreach.hpp>

#include <fast_filtering/utils/helper_functions.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>
#include <pose_tracking_interface/utils/pcl_interface.hpp>
#include <pose_tracking_interface/utils/robot_tracking_dataset.hpp>

RobotTrackingDataset::RobotTrackingDataset(const std::string& path) : TrackingDataset(path), 
				       ground_truth_joints_topic_("joint_states"),
				       noisy_joints_topic_("noisy_joint_states"),
				       deviation_filename_("deviation.txt") {}
			    
void RobotTrackingDataset::AddFrame(
        const sensor_msgs::Image::ConstPtr& image,
        const sensor_msgs::CameraInfo::ConstPtr& info,
	const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
	const sensor_msgs::JointState::ConstPtr& noisy_joints,
        const Eigen::VectorXd& ground_truth,
        const Eigen::VectorXd& deviation)
{
  DataFrame data(image, info, ground_truth_joints, noisy_joints, ground_truth, deviation);
  data_.push_back(data);
}

void RobotTrackingDataset::AddFrame(
        const sensor_msgs::Image::ConstPtr& image,
        const sensor_msgs::CameraInfo::ConstPtr& info,
	const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
	const sensor_msgs::JointState::ConstPtr& noisy_joints)
{
  DataFrame data(image, info, ground_truth_joints, noisy_joints);
  data_.push_back(data);
}

Eigen::VectorXd RobotTrackingDataset::GetDeviation(const size_t& index)
{
  return data_[index].deviation_;
}

sensor_msgs::JointState::ConstPtr RobotTrackingDataset::GetGroundTruthJoints(const size_t& index)
{
    return data_[index].ground_truth_joints_;
}

sensor_msgs::JointState::ConstPtr RobotTrackingDataset::GetNoisyJoints(const size_t& index)
{
    return data_[index].noisy_joints_;
}

void RobotTrackingDataset::Load()
{
    // load bagfile ----------------------------------------------------------------------------
    rosbag::Bag bag;
    bag.open((path_ / observations_filename_).string(), rosbag::bagmode::Read);

    // Image topics to load
    std::vector<std::string> topics;
    topics.push_back(image_topic_);
    topics.push_back(info_topic_);
    topics.push_back(ground_truth_joints_topic_);
    topics.push_back(noisy_joints_topic_);
    topics.push_back("/" + image_topic_);
    topics.push_back("/" + info_topic_);
    topics.push_back("/" + ground_truth_joints_topic_);
    topics.push_back("/" + noisy_joints_topic_);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Set up fake subscribers to capture images
    BagSubscriber<sensor_msgs::Image> image_subscriber;
    BagSubscriber<sensor_msgs::CameraInfo> info_subscriber;
    BagSubscriber<sensor_msgs::JointState> ground_truth_subscriber;
    BagSubscriber<sensor_msgs::JointState> noisy_subscriber;

    // Use time synchronizer to make sure we get properly synchronized images
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::JointState, sensor_msgs::JointState>
      sync(image_subscriber, info_subscriber, ground_truth_subscriber, noisy_subscriber, 25);
    sync.registerCallback(boost::bind(&RobotTrackingDataset::AddFrame, this,  _1, _2, _3, _4));

    // Load all messages into our stereo TrackingDataset
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
        if (m.getTopic() == image_topic_ || (m.getTopic() == "/" + image_topic_))
        {
            sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
            if (image != NULL)
                image_subscriber.newMessage(image);
        }

        if (m.getTopic() == info_topic_ || (m.getTopic() == "/" + info_topic_))
        {
            sensor_msgs::CameraInfo::ConstPtr info = m.instantiate<sensor_msgs::CameraInfo>();
            if (info != NULL)
                info_subscriber.newMessage(info);
        }
	
	if (m.getTopic() == ground_truth_joints_topic_ || (m.getTopic() == "/" + ground_truth_joints_topic_))
	  {
            sensor_msgs::JointState::ConstPtr gt_joints = m.instantiate<sensor_msgs::JointState>();
            if (gt_joints != NULL)
	      ground_truth_subscriber.newMessage(gt_joints);
	  }
	
	if (m.getTopic() == noisy_joints_topic_ || (m.getTopic() == "/" + noisy_joints_topic_))
	  {
            sensor_msgs::JointState::ConstPtr noisy_joints = m.instantiate<sensor_msgs::JointState>();
            if (noisy_joints != NULL)
	      noisy_subscriber.newMessage(noisy_joints);
	  }
    }
    bag.close();

    // load ground_truth.txt ---------------------------------------------------------------------
    if (! LoadTextFile((path_ / ground_truth_filename_).c_str(), DataType::GROUND_TRUTH))
      std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
    exit(-1);
    
    // load deviation file   ---------------------------------------------------------------------
    if (! LoadTextFile((path_ / deviation_filename_).c_str(), DataType::DEVIATION))
      std::cout << "could not open file " << path_ / deviation_filename_ << std::endl;
    exit(-1);
}


void RobotTrackingDataset::Store()
{
    if(boost::filesystem::exists(path_ / observations_filename_) ||
       boost::filesystem::exists(path_ / ground_truth_filename_) || 
       boost::filesystem::exists(path_ / deviation_filename_))
    {
        std::cout << "TrackingDataset with name " << path_ << " already exists, will not overwrite." << std::endl;
        return;
    }
    else
        boost::filesystem::create_directory(path_);

    // write images to bagfile -----------------------------------------------------------------
    rosbag::Bag bag;
    bag.open((path_ / observations_filename_).string(), rosbag::bagmode::Write);

    for(size_t i = 0; i < data_.size(); i++)
    {
        bag.write(image_topic_, data_[i].image_->header.stamp, data_[i].image_);
        bag.write(info_topic_, data_[i].info_->header.stamp, data_[i].info_);
        bag.write(ground_truth_joints_topic_, data_[i].ground_truth_joints_->header.stamp, data_[i].ground_truth_joints_);
        bag.write(noisy_joints_topic_, data_[i].noisy_joints_->header.stamp, data_[i].noisy_joints_);
    }
    bag.close();

    // write ground truth to txt file ----------------------------------------------------------
    if(!StoreTextFile((path_ / ground_truth_filename_).c_str(), DataType::GROUND_TRUTH)) 
      {
	std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
	exit(-1);
      }

    // write deviation to txt file ----------------------------------------------------------
    if(!StoreTextFile((path_ / deviation_filename_).c_str(), DataType::DEVIATION)) 
      {
	std::cout << "could not open file " << path_ / deviation_filename_ << std::endl;
	exit(-1);
      }

}
