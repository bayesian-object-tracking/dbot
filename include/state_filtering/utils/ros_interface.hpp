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

#ifndef POSE_FILTERING_ROS_INTERFACE_HPP_
#define POSE_FILTERING_ROS_INTERFACE_HPP_

#include <string>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/CameraInfo.h>

#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


//#include <cv.h>

// to avoid stupid typedef conflict
#define uint64 enchiladisima
#include <cv_bridge/cv_bridge.h>
#undef uint64

#include <sensor_msgs/Image.h>


namespace ri
{


template<typename Parameter>
void ReadParameter(const std::string& path,
                   Parameter& parameter,
                   ros::NodeHandle node_handle)
{
    XmlRpc::XmlRpcValue ros_parameter;
    node_handle.getParam(path, ros_parameter);
    parameter = Parameter(ros_parameter);
}

template<>
void ReadParameter<std::vector<std::string> >(const std::string& path,
                                              std::vector<std::string>& parameter,
                                              ros::NodeHandle node_handle);

template<>
void ReadParameter<std::vector<double> >(const std::string& path,
                                         std::vector<double>& parameter,
                                         ros::NodeHandle node_handle);

template<>
void ReadParameter<std::vector<std::vector<size_t> > >(const std::string& path,
                                                       std::vector<std::vector<size_t> >& parameter,
                                                       ros::NodeHandle node_handle);

template<typename Scalar> Eigen::Matrix<Scalar, -1, -1>
Ros2Eigen(const sensor_msgs::Image& ros_image,
          const size_t& n_downsampling = 1)
{

    cv::Mat cv_image = cv_bridge::toCvCopy(ros_image)->image;

    size_t n_rows = cv_image.rows/n_downsampling; size_t n_cols = cv_image.cols/n_downsampling;
    Eigen::Matrix<Scalar, -1, -1> eigen_image(n_rows, n_cols);
    for(size_t row = 0; row < n_rows; row++)
        for(size_t col = 0; col < n_cols; col++)
            eigen_image(row, col) = cv_image.at<float>(row*n_downsampling,col*n_downsampling);

    return eigen_image;
}

template<typename Scalar> Eigen::Matrix<Scalar, 3, 3>
GetCameraMatrix(const std::string& camera_info_topic,
                ros::NodeHandle& node_handle,
                const Scalar& seconds)
{
  // TODO: Check if const pointer is valid before accessing memory
    sensor_msgs::CameraInfo::ConstPtr camera_info =
            ros::topic::waitForMessage<sensor_msgs::CameraInfo> (camera_info_topic,
                                                                 node_handle,
                                                                 ros::Duration(seconds));
   
    Eigen::Matrix<Scalar, 3, 3> camera_matrix = Eigen::Matrix<Scalar, 3, 3>::Zero();
    
    if(!camera_info) {
      // if not topic was received within <seconds>
      ROS_WARN("CameraInfo wasn't received within %f seconds. Returning default Zero message.", seconds);
      return camera_matrix;
    }
    ROS_INFO("Valid CameraInfo was received");
    
    for(size_t col = 0; col < 3; col++)
        for(size_t row = 0; row < 3; row++)
	  camera_matrix(row,col) = camera_info->K[col+row*3];

    return camera_matrix;
}


void PublishMarker(const Eigen::Matrix3f R, const Eigen::Vector3f t,
		std_msgs::Header header,
		std::string object_model_path,
		const ros::Publisher &pub,
        int marker_id = 0, float r = 0, float g = 1, float b = 0, float a = 1.0);

void PublishMarker(const Eigen::Matrix4f H,
		std_msgs::Header header,
		std::string object_model_path,
		const ros::Publisher &pub,
        int marker_id = 0, float r = 0, float g = 1, float b = 0, float a = 1.0);



void PublishPoints(const std_msgs::Header header,
		const ros::Publisher &pub,
        const std::vector<Eigen::Vector3f> points,
        std::vector<float> colors = std::vector<float>(0),
        const Eigen::Matrix3f R = Eigen::Matrix3f::Identity(),
        Eigen::Vector3f t = Eigen::Vector3f::Zero());

}



#endif
