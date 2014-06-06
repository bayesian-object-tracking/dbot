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

#ifndef POSE_FILTERING_ROS_INTERFACE_HPP_
#define POSE_FILTERING_ROS_INTERFACE_HPP_


#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ros/conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
void ReadParameter< std::vector<std::string> >(const std::string& path,
                                               std::vector<std::string>& parameter,
                                               ros::NodeHandle node_handle)
{
    XmlRpc::XmlRpcValue ros_parameter;
    node_handle.getParam(path, ros_parameter);
    parameter.resize(ros_parameter.size());
    for(size_t i = 0; i < parameter.size(); i++)
        parameter[i] = std::string(ros_parameter[i]);
}

template<>
void ReadParameter< std::vector<std::vector<size_t> > >(const std::string& path,
                                                        std::vector<std::vector<size_t> >& parameter,
                                                        ros::NodeHandle node_handle)
{
    XmlRpc::XmlRpcValue ros_parameter;
    node_handle.getParam(path, ros_parameter);
    parameter.resize(ros_parameter.size());
    for(size_t i = 0; i < parameter.size(); i++)
    {
        parameter[i].resize(ros_parameter[i].size());
        for(size_t j = 0; j < parameter[i].size(); j++)
            parameter[i][j] = int(ros_parameter[i][j]);
    }
}



template<typename Scalar> Eigen::Matrix<Scalar, 3, 3>
GetCameraMatrix(const std::string& camera_info_topic,
                ros::NodeHandle& node_handle,
                const Scalar& seconds)
{
    sensor_msgs::CameraInfo::ConstPtr camera_info =
            ros::topic::waitForMessage<sensor_msgs::CameraInfo> (camera_info_topic,
                                                                 node_handle,
                                                                 ros::Duration(seconds));
    Eigen::Matrix<Scalar, 3, 3> camera_matrix;
    for(size_t col = 0; col < 3; col++)
        for(size_t row = 0; row < 3; row++)
            camera_matrix(row,col) = camera_info->K[col+row*3];

    return camera_matrix;
}


void PublishMarker(const Eigen::Matrix3f R, const Eigen::Vector3f t,
		std_msgs::Header header,
		std::string object_model_path,
		const ros::Publisher &pub,
		int marker_id = 0, float r = 0, float g = 1, float b = 0, float a = 1.0)
{

	Eigen::Quaternion<float> q(R);

	geometry_msgs::PoseWithCovarianceStamped pose;
	pose.header =  header;
	pose.pose.pose.position.x = t(0);
	pose.pose.pose.position.y = t(1);
	pose.pose.pose.position.z = t(2);

	pose.pose.pose.orientation.x = q.x();
	pose.pose.pose.orientation.y = q.y();
	pose.pose.pose.orientation.z = q.z();
	pose.pose.pose.orientation.w = q.w();


	visualization_msgs::Marker marker;
	marker.header.frame_id =  pose.header.frame_id;
	marker.header.stamp =  pose.header.stamp;
	marker.ns = "object_pose_estimation";
	marker.id = marker_id;
	marker.scale.x = 1.0;
	marker.scale.y = 1.0;
	marker.scale.z = 1.0;
	marker.color.r = r;
	marker.color.g = g;
	marker.color.b = b;
	marker.color.a = a;

	marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose = pose.pose.pose;

	marker.mesh_resource = object_model_path;

	pub.publish(marker);
}
void PublishMarker(const Eigen::Matrix4f H,
		std_msgs::Header header,
		std::string object_model_path,
		const ros::Publisher &pub,
		int marker_id = 0, float r = 0, float g = 1, float b = 0, float a = 1.0)
{
    PublishMarker(H.topLeftCorner(3,3),
                  H.topRightCorner(3,1),
                  header,
                  object_model_path, pub,
                  marker_id, r, g, b, a);

}





void PublishPoints(const std_msgs::Header header,
		const ros::Publisher &pub,
		const std::vector<Eigen::Vector3f> points, std::vector<float> colors = std::vector<float>(0),
		const Eigen::Matrix3f R = Eigen::Matrix3f::Identity(), Eigen::Vector3f t = Eigen::Vector3f::Zero())
{
	// if no color has been given we set it to some value -----------------------------
	if(colors.size() == 0)
		colors = std::vector<float> (points.size(), 1);

	// renormalize colors -----------------------------
	float max = -std::numeric_limits<float>::max();
	float min = std::numeric_limits<float>::max();
	for(int i = 0; i < int(colors.size()); i++)
	{
		min = colors[i] < min ? colors[i] : min;
		max = colors[i] > max ? colors[i] : max;
	}
	if(min == max) min = 0;

	pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
	point_cloud.header = header;
	point_cloud.width    = points.size(); point_cloud.height   = 1; point_cloud.is_dense = false;
	point_cloud.points.resize (point_cloud.width * point_cloud.height);

	for (int point_index = 0; point_index < int(points.size()); ++point_index)
	{
		Eigen::Vector3f new_point = R*points[point_index] + t;
		point_cloud.points[point_index].x = new_point(0);
		point_cloud.points[point_index].y = new_point(1);
		point_cloud.points[point_index].z = new_point(2);

		point_cloud.points[point_index].r = (colors[point_index]-min)/(max-min) * 255.;
		point_cloud.points[point_index].g = 0.;
		point_cloud.points[point_index].b = (1 - (colors[point_index]-min)/(max-min)) * 255.;
	}
	sensor_msgs::PointCloud2 point_cloud2;
	pcl::toROSMsg(point_cloud, point_cloud2);
	pub.publish(point_cloud2);
}

}



#endif
