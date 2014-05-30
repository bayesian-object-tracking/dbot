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

int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_filter");
    ros::NodeHandle node_handle("~");

    // read params --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    string point_cloud_topic; ReadParam("point_cloud_topic", point_cloud_topic, node_handle);
    string camera_info_topic; ReadParam("camera_info_topic", camera_info_topic, node_handle);
    int initial_sample_count; ReadParam("initial_sample_count", initial_sample_count, node_handle);

    // read camera_matrix ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cout << "reading camera matrix" << endl;
    sensor_msgs::CameraInfo::ConstPtr camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>
            (camera_info_topic, node_handle, ros::Duration(2.0));
    Matrix3d camera_matrix;
    for(unsigned int col = 0; col < 3; col++)
        for(unsigned int row = 0; row < 3; row++)
            camera_matrix(row,col) = camera_info->K[col+row*3];
    cout << camera_matrix << endl;

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

    // find points on table and cluster them ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<Vector3d> table_points;
    size_t table_rows, table_cols;
    Vector4d table_plane;
    pi::PointsOnPlane(all_points, all_rows, all_cols, table_points, table_rows, table_cols, table_plane, true);
    if(table_plane.topRows(3).dot(Eigen::Vector3d(0,1,0)) > 0)
        table_plane = - table_plane;
    table_plane /= table_plane.topRows(3).norm();
    Vector3d table_normal = table_plane.topRows(3);
    vector<vector<Vector3d> > clusters;
    pi::Cluster(table_points, table_rows, table_cols, clusters);
    if(clusters.size() == 0)
    {
        cout << "no objects found on table " << endl;
        return 0;
    }

    // we create samples around the clusters on the table --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<VectorXd> initial_states;
    // create gaussian for sampling
    double standard_deviation_translation = 0.03;
    double standard_deviation_rotation = 100.0;
    GaussianDistribution<double, 1> unit_gaussian;
    unit_gaussian.setNormal();
    //unit_gaussian.mean(GaussianDistribution<double, 1>::VariableType::Zero());
    //unit_gaussian.covariance(GaussianDistribution<double, 1>::CovarianceType::Identity());

    cout << "found " << clusters.size() << " clusters on table, we will sample around each cluster" 	<< endl;
    for(size_t cluster_index = 0; cluster_index < clusters.size(); cluster_index++)
    {
        Vector3d com(0,0,0);
        for(unsigned int i = 0; i < clusters[cluster_index].size(); i++)
            com += clusters[cluster_index][i];
        com /= float(clusters[cluster_index].size());

        Vector3d t_mean = com - (com.dot(table_normal)+table_plane(3))*table_normal; // project center of mass in table plane
        Vector3d table_vector_a = table_normal.cross(Vector3d(1,1,1)).normalized(); // vector along table plane
        Vector3d table_vector_b = table_normal.cross(table_vector_a); // second vector along table plane
        Matrix3d R_mean; R_mean.col(0) = table_vector_a; R_mean.col(1) = table_vector_b; R_mean.col(2) = table_normal;

        // sample around mean
        for(size_t i = 0; i < size_t(initial_sample_count)/clusters.size(); i++)
        {
            FullRigidBodySystem<-1> state(1);
            state.translation() =
                    t_mean +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_a +
                    standard_deviation_translation * unit_gaussian.sample()(0) * table_vector_b;
            state.orientation() = Quaterniond(
                        AngleAxisd(standard_deviation_rotation * unit_gaussian.sample()(0), table_normal) * R_mean).coeffs();

            initial_states.push_back(state);
        }
    }

    // intialize the filter ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    TestFilter test_filter(camera_matrix);
    test_filter.Initialize(initial_states, ros_cloud);
    cout << "done initializing" << endl;

    ros::Subscriber subscriber =
            node_handle.subscribe(point_cloud_topic, 1, &TestFilter::Filter, &test_filter);

    ros::spin();
    return 0;
}
