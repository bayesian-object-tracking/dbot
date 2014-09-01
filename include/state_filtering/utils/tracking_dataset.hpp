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

#ifndef TRACKING_DATASET_HPP_
#define TRACKING_DATASET_HPP_

#include <Eigen/Dense>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/simple_filter.h>

#include <fstream>

#include <boost/filesystem.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//#include <state_filtering/utils/helper_functions.hpp>
//#include <state_filtering/utils/ros_interface.hpp>
//#include <state_filtering/utils/pcl_interface.hpp>

class DataFrame
{
public:
    sensor_msgs::Image::ConstPtr image_;
    sensor_msgs::CameraInfo::ConstPtr info_;
    Eigen::VectorXd ground_truth_;

    DataFrame(const sensor_msgs::Image::ConstPtr& image,
              const sensor_msgs::CameraInfo::ConstPtr& info,
              const Eigen::VectorXd& ground_truth = Eigen::VectorXd());
};

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const> &msg)
    {
      this->signalMessage(msg);
    }
};


class TrackingDataset
{
public:
    TrackingDataset(const std::string& path);
    ~TrackingDataset();

    void addFrame(const sensor_msgs::Image::ConstPtr& image,
                  const sensor_msgs::CameraInfo::ConstPtr& info,
                  const Eigen::VectorXd& ground_truth = Eigen::VectorXd());

    void addFrame(const sensor_msgs::Image::ConstPtr& image,
                  const sensor_msgs::CameraInfo::ConstPtr& info);

    sensor_msgs::Image::ConstPtr getImage(const size_t& index);

    sensor_msgs::CameraInfo::ConstPtr getInfo(const size_t& index);

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr getPointCloud(const size_t& index);

    Eigen::Matrix3d getCameraMatrix(const size_t& index);

    Eigen::VectorXd getGroundTruth(const size_t& index);

    size_t sIze();

    void loAd();

    void stOre();

private:
    std::vector<DataFrame> data_;

    const boost::filesystem::path path_;
    const std::string image_topic_;
    const std::string info_topic_;
    const std::string observations_filename_;
    const std::string ground_truth_filename_;
    const double admissible_delta_time_; // admissible time difference in s for comparing time stamps
};

#endif
