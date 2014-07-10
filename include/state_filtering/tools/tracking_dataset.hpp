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

#ifndef TRACKING_DATASET_HPP_
#define TRACKING_DATASET_HPP_

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <state_filtering/tools/helper_functions.hpp>
#include <state_filtering/tools/ros_interface.hpp>
#include <state_filtering/tools/pcl_interface.hpp>










class DataFrame
{
public:
    sensor_msgs::Image::ConstPtr image_;
    sensor_msgs::CameraInfo::ConstPtr info_;
    Eigen::VectorXd ground_truth_;


    DataFrame(const sensor_msgs::Image::ConstPtr& image,
              const sensor_msgs::CameraInfo::ConstPtr& info,
              const Eigen::VectorXd& ground_truth = Eigen::VectorXd()):
        image_(image),
        info_(info),
        ground_truth_(ground_truth){ }
};

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const> &msg)
    {
        signalMessage(msg);
    }
};


class TrackingDataset
{
public:
    TrackingDataset(const std::string& path):path_(path),
                                image_topic_("XTION/depth/image"),
                                info_topic_("XTION/depth/camera_info"),
                                measurements_filename_("measurements.bag"),
                                ground_truth_filename_("ground_truth.txt"),
                                admissible_delta_time_(0.0001) {}
    ~TrackingDataset() {}

    void addFrame(const sensor_msgs::Image::ConstPtr& image,
                  const sensor_msgs::CameraInfo::ConstPtr& info,
                   const Eigen::VectorXd& ground_truth = Eigen::VectorXd())
    {
        DataFrame data(image, info, ground_truth);
        data_.push_back(data);
    }

    void addFrame(const sensor_msgs::Image::ConstPtr& image,
                  const sensor_msgs::CameraInfo::ConstPtr& info)
    {
        DataFrame data(image, info);
        data_.push_back(data);
    }

    sensor_msgs::Image::ConstPtr getImage(const size_t& index)
    {
        return data_[index].image_;
    }

    sensor_msgs::CameraInfo::ConstPtr getInfo(const size_t& index)
    {
        return data_[index].info_;
    }
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr getPointCloud(const size_t& index)
    {
        Eigen::MatrixXd image = ri::Ros2Eigen<double>(*data_[index].image_);
        Eigen::Matrix<Eigen::Matrix<double, 3, 1> , -1, -1> points = hf::Image2Points(image, getCameraMatrix(index));
        pcl::PointCloud<pcl::PointXYZ>::Ptr
                point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>));

        point_cloud->header = data_[index].image_->header;
        pi::Eigen2Pcl(points, *point_cloud);

        return point_cloud;
    }

    Eigen::Matrix3d getCameraMatrix(const size_t& index)
    {
        Eigen::Matrix3d camera_matrix;
        for(size_t col = 0; col < 3; col++)
            for(size_t row = 0; row < 3; row++)
                camera_matrix(row,col) = data_[0].info_->K[col+row*3];
        return camera_matrix;
    }

    Eigen::VectorXd getGroundTruth(const size_t& index)
    {
        return data_[index].ground_truth_;
    }

    size_t sIze()
    {
        return data_.size();
    }

    void loAd()
    {
        // load bagfile ----------------------------------------------------------------------------
        rosbag::Bag bag;
        bag.open((path_ / measurements_filename_).string(), rosbag::bagmode::Read);

        // Image topics to load
        std::vector<std::string> topics;
        topics.push_back(image_topic_);
        topics.push_back(info_topic_);
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // Set up fake subscribers to capture images
        BagSubscriber<sensor_msgs::Image> image_subscriber;
        BagSubscriber<sensor_msgs::CameraInfo> info_subscriber;

        // Use time synchronizer to make sure we get properly synchronized images
        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo>
                sync(image_subscriber, info_subscriber, 25);
        sync.registerCallback(boost::bind(&TrackingDataset::addFrame, this,  _1, _2));

        // Load all messages into our stereo TrackingDataset
        BOOST_FOREACH(rosbag::MessageInstance const m, view)
        {
            if (m.getTopic() == image_topic_ || ("/" + m.getTopic() == image_topic_))
            {
                sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
                if (image != NULL)
                    image_subscriber.newMessage(image);
            }

            if (m.getTopic() == info_topic_ || ("/" + m.getTopic() == info_topic_))
            {
                sensor_msgs::CameraInfo::ConstPtr info = m.instantiate<sensor_msgs::CameraInfo>();
                if (info != NULL)
                    info_subscriber.newMessage(info);
            }
        }
        bag.close();

        // load ground_truth.txt ---------------------------------------------------------------------
        std::ifstream file; file.open((path_ / ground_truth_filename_).c_str(), std::ios::in); // open file
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
            file.close();

            // attach the state to the appropriate data frames
            for(size_t i = 0; i < data_.size(); i++)
                if(std::fabs(data_[i].image_->header.stamp.toSec() - time_stamp) <= admissible_delta_time_)
                    data_[i].ground_truth_ = state;
        }
        else
        {
            std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
            exit(-1);
        }
    }

    void stOre()
    {
        if(boost::filesystem::exists(path_))
        {
//            std::cout << "TrackingDataset with name " << path_ << " already exists, will not overwrite." << std::endl;
//            return;
        }
        else
            boost::filesystem::create_directory(path_);

        // write images to bagfile -----------------------------------------------------------------
        rosbag::Bag bag;
        bag.open((path_ / measurements_filename_).string(), rosbag::bagmode::Write);

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
        std::ofstream file; file.open((path_ / ground_truth_filename_).c_str(), std::ios::out | std::ios::trunc);
        if(file.is_open())
        {
            for(size_t i = 0; i < data_.size(); i++)
            {
                if(data_[i].ground_truth_.rows() != 0)
                {
                    file << data_[i].image_->header.stamp << " ";
                    file << data_[i].ground_truth_.transpose() << std::endl;
                }
            }
            file.close();
        }
        else
        {
            std::cout << "could not open file " << path_ / ground_truth_filename_ << std::endl;
            exit(-1);
        }
    }

private:
    std::vector<DataFrame> data_;

    const boost::filesystem::path path_;
    const std::string image_topic_;
    const std::string info_topic_;
    const std::string measurements_filename_;
    const std::string ground_truth_filename_;
    const double admissible_delta_time_; // admissible time difference in s for comparing time stamps
};

















#endif
