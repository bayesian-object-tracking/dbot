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

#include <state_filtering/run_cpu_coordinate_filter.hpp>

typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;

int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_filter");
    ros::NodeHandle node_handle("~");

    // read parameters
    string point_cloud_topic; ri::ReadParameter("point_cloud_topic", point_cloud_topic, node_handle);
    string camera_info_topic; ri::ReadParameter("camera_info_topic", camera_info_topic, node_handle);
    int initial_sample_count; ri::ReadParameter("initial_sample_count", initial_sample_count, node_handle);

    Matrix3d camera_matrix = ri::GetCameraMatrix<double>(camera_info_topic, node_handle, 2.0);

    // get observations from camera ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cout << "reading point cloud " << endl;
    sensor_msgs::PointCloud2 ros_cloud  =
            *ros::topic::waitForMessage<sensor_msgs::PointCloud2>(point_cloud_topic, node_handle, ros::Duration(2.0));
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg (ros_cloud, *pcl_cloud);
    cout << "done" << endl;
    vector<Vector3d> all_points;
    size_t all_rows, all_cols;
    pi::Pcl2Eigen(*pcl_cloud, all_points, all_rows, all_cols);

   vector<VectorXd> initial_states = pi::SampleTableClusters(all_points, all_rows, all_cols, initial_sample_count);

    // intialize the filter ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    RunCpuCoordinateFilter test_filter(camera_matrix);
    test_filter.Initialize(initial_states, ros_cloud);
    cout << "done initializing" << endl;

    ros::Subscriber subscriber =
            node_handle.subscribe(point_cloud_topic, 1, &RunCpuCoordinateFilter::Filter, &test_filter);

    ros::spin();
    return 0;
}
