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
#include <pose_tracking_interface/utils/tracking_dataset.hpp>


DataFrame::DataFrame(const sensor_msgs::Image::ConstPtr& image,
		     const sensor_msgs::CameraInfo::ConstPtr& info,
		     const Eigen::VectorXd& ground_truth,
		     const Eigen::VectorXd& deviation):
    image_(image),
    info_(info),
    ground_truth_(ground_truth),
    deviation_(deviation) { }

DataFrame::DataFrame(const sensor_msgs::Image::ConstPtr& image,
          const sensor_msgs::CameraInfo::ConstPtr& info,
	  const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
	  const sensor_msgs::JointState::ConstPtr& noisy_joints,
          const Eigen::VectorXd& ground_truth,
          const Eigen::VectorXd& deviation):
    image_(image),
    info_(info),
    ground_truth_joints_(ground_truth_joints),
    noisy_joints_(noisy_joints),
    ground_truth_(ground_truth),
    deviation_(deviation) { }

DataFrame::DataFrame(const sensor_msgs::Image::ConstPtr& image,
		     const sensor_msgs::CameraInfo::ConstPtr& info,
		     const sensor_msgs::JointState::ConstPtr& ground_truth_joints,
		     const sensor_msgs::JointState::ConstPtr& noisy_joints,
		     const tf::tfMessage::ConstPtr& tf,
		     const tf::tfMessage::ConstPtr& fixed_tf,
		     const Eigen::VectorXd& ground_truth,
		     const Eigen::VectorXd& deviation):
  image_(image),
  info_(info),
  ground_truth_joints_(ground_truth_joints),
  noisy_joints_(noisy_joints),
  gt_tf_(tf),
  gt_tf_fixed_(fixed_tf),
  ground_truth_(ground_truth),
  deviation_(deviation) { }

TrackingDataset::TrackingDataset(const std::string& path):path_(path),
                            image_topic_("XTION/depth/image"),
                            info_topic_("XTION/depth/camera_info"),
                            observations_filename_("measurements.bag"),
                            ground_truth_filename_("ground_truth.txt"),
                            admissible_delta_time_(0.02) {}

TrackingDataset::~TrackingDataset() {}

void TrackingDataset::AddFrame(
        const sensor_msgs::Image::ConstPtr& image,
        const sensor_msgs::CameraInfo::ConstPtr& info,
        const Eigen::VectorXd& ground_truth,
        const Eigen::VectorXd& deviation)
{
  DataFrame data(image, info, ground_truth,deviation);
    data_.push_back(data);
}

void TrackingDataset::AddFrame(const sensor_msgs::Image::ConstPtr& image,
                               const sensor_msgs::CameraInfo::ConstPtr& info)
{
    DataFrame data(image, info);
    data_.push_back(data);
}

sensor_msgs::Image::ConstPtr TrackingDataset::GetImage(const size_t& index)
{
    return data_[index].image_;
}

sensor_msgs::CameraInfo::ConstPtr TrackingDataset::GetInfo(const size_t& index)
{
    return data_[index].info_;
}
pcl::PointCloud<pcl::PointXYZ>::ConstPtr TrackingDataset::GetPointCloud(
        const size_t& index)
{
    Eigen::MatrixXd image = ri::Ros2Eigen<double>(*data_[index].image_);
    Eigen::Matrix<Eigen::Matrix<double, 3, 1> , -1, -1> points = ff::hf::Image2Points(image, GetCameraMatrix(index));
    pcl::PointCloud<pcl::PointXYZ>::Ptr
            point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>));

    point_cloud->header = data_[index].image_->header;
    pi::Eigen2Pcl(points, *point_cloud);

    return point_cloud;
}

Eigen::Matrix3d TrackingDataset::GetCameraMatrix(const size_t& index)
{
    Eigen::Matrix3d camera_matrix;
    for(size_t col = 0; col < 3; col++)
        for(size_t row = 0; row < 3; row++)
            camera_matrix(row,col) = data_[0].info_->K[col+row*3];
    return camera_matrix;
}

Eigen::VectorXd TrackingDataset::GetGroundTruth(const size_t& index)
{
    return data_[index].ground_truth_;
}

size_t TrackingDataset::Size()
{
    return data_.size();
}

void TrackingDataset::Load()
{
    // load bagfile ----------------------------------------------------------------------------
    rosbag::Bag bag;
    bag.open((path_ / observations_filename_).string(), rosbag::bagmode::Read);

    // Image topics to load
    std::vector<std::string> topics;
    topics.push_back(image_topic_);
    topics.push_back(info_topic_);
    topics.push_back("/" + image_topic_);
    topics.push_back("/" + info_topic_);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Set up fake subscribers to capture images
    BagSubscriber<sensor_msgs::Image> image_subscriber;
    BagSubscriber<sensor_msgs::CameraInfo> info_subscriber;

    // Use time synchronizer to make sure we get properly synchronized images
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo>
            sync(image_subscriber, info_subscriber, 25);
    sync.registerCallback(boost::bind(&TrackingDataset::AddFrame, this,  _1, _2));

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
    }
    bag.close();

    // load ground_truth.txt ---------------------------------------------------------------------
    if (! LoadTextFile((path_ / ground_truth_filename_).c_str(), DataType::GROUND_TRUTH))
      std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
    exit(-1);
    
}

bool TrackingDataset::LoadTextFile(const char *filename, DataType type)
{
  std::ifstream file; file.open(filename, std::ios::in); // open file
  if(file.is_open())
    {
      std::string temp; std::getline(file, temp); std::istringstream line(temp); // get a line from file
      double time_stamp; line >> time_stamp; // get the timestamp
      double scalar; Eigen::VectorXd state;
      while(line >> scalar) // get the state
        {
	  Eigen::VectorXd temp(state.rows() + 1);
	  temp.topRows(state.rows()) = state;
	  temp(temp.rows()-1) = scalar;
	  state = temp;
        }
      
      std::cout << "read state " << state.transpose() << std::endl;
      std::cout << "read bagfile of size " << data_.size() << std::endl;
      std::cout << "timestamp of first image is " << data_[0].image_->header.stamp << std::endl;
      file.close();
      
      // attach the state to the appropriate data frames
      for(size_t i = 0; i < data_.size(); i++)
	if(std::fabs(data_[i].image_->header.stamp.toSec() - time_stamp) <= admissible_delta_time_)
	  switch (type)
	    {
	    case DataType::GROUND_TRUTH:
	      data_[i].ground_truth_ = state;
	      break;
	    case DataType::DEVIATION:
	      data_[i].deviation_ = state;
	      break;
	    default:
	      return false;
	    }
      return true;
    }
  else return false;
    {
      std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
      exit(-1);
    }

}

void TrackingDataset::Store()
{
    if(boost::filesystem::exists(path_ / observations_filename_) ||
       boost::filesystem::exists(path_ / ground_truth_filename_))
    {
        std::cout << "TrackingDataset with name " << path_ << " already exists, will not overwrite." << std::endl;
        return;
    }
    else
        boost::filesystem::create_directory(path_);

    // write images to bagfile -----------------------------------------------------------------
    rosbag::Bag bag;
    bag.open((path_ / observations_filename_).string(), rosbag::bagmode::Write);

    std::vector<std::string> topics;
    topics.push_back(image_topic_);
    topics.push_back(info_topic_);

    for(size_t i = 0; i < data_.size(); i++)
    {
        bag.write(image_topic_, data_[i].image_->header.stamp, data_[i].image_);
        bag.write(info_topic_, data_[i].info_->header.stamp, data_[i].info_);
    }
    bag.close();

    // write ground truth to txt file ----------------------------------------------------------
    if(!StoreTextFile((path_ / ground_truth_filename_).c_str(), DataType::GROUND_TRUTH)) 
      {
	std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
	exit(-1);
      }
}


bool TrackingDataset::StoreTextFile(const char *filename, DataType type)
{
  std::ofstream file; file.open(filename, std::ios::out | std::ios::trunc);
  if(file.is_open())
    {
      for(size_t i = 0; i < data_.size(); i++)
        {
	  switch(type) 
	    {
	    case(DataType::GROUND_TRUTH):
	      if(data_[i].ground_truth_.rows() != 0)
		{
		  file << data_[i].image_->header.stamp << " ";
		  file << data_[i].ground_truth_.transpose() << std::endl;
		}
	      break;
	    case(DataType::DEVIATION):
	      if(data_[i].deviation_.rows() != 0)
		{
		  file << data_[i].image_->header.stamp << " ";
		  file << data_[i].deviation_.transpose() << std::endl;
		}
	      break;
	    default:
	      return false;
	    }
        }
      file.close();
    }
  else return false;
  
  return true;
}

