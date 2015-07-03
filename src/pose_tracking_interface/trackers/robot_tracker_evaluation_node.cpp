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


#include <fstream>
#include <ctime>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include <boost/filesystem.hpp>

#include <dbot/utils/profiling.hpp>


#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

#include <pose_tracking_interface/trackers/robot_tracker.hpp>
#include <pose_tracking_interface/utils/cloud_visualizer.hpp>
#include <pose_tracking_interface/utils/kinematics_from_urdf.hpp>

#include <pose_tracking_interface/utils/ros_interface.hpp>

#include <cv.h>
#include <cv_bridge/cv_bridge.h>


#include <pose_tracking_interface/trackers/object_tracker.hpp>
#include <pose_tracking_interface/utils/tracking_dataset.hpp>
#include <pose_tracking_interface/utils/pcl_interface.hpp>
#include <pose_tracking_interface/utils/ros_interface.hpp>

//#include <dbot/utils/distribution_test.hpp>

typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;
typedef Eigen::Matrix<double, -1, -1> Image;

class TrackerInterface
{
public:
    TrackerInterface(boost::shared_ptr<RobotTracker> tracker): tracker_(tracker), node_handle_("~")
    {
        std::string config_file;
        ri::ReadParameter("config_file", config_file, node_handle_);

        path_ = config_file;
        path_ = path_.parent_path();
        std::cout << path_ << std::endl;

        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,80,"%d.%m.%Y_%I.%M.%S",timeinfo);
        std::string current_time(buffer);

        path_ /= "tracking_data_" + current_time + ".txt";
    }
    ~TrackerInterface() {}

    void Filter(const sensor_msgs::Image& ros_image)
    {
        INIT_PROFILING
        Eigen::VectorXd mean_state = tracker_->FilterAndReturn(ros_image);
        MEASURE("total time for filtering")
    }

    void FilterAndStore(const sensor_msgs::Image& ros_image)
    {
        INIT_PROFILING
        Eigen::VectorXd mean_state = tracker_->FilterAndReturn(ros_image);
        MEASURE("total time for filtering")

        std::ofstream file;
        file.open(path_.c_str(), std::ios::out | std::ios::app);
        if(file.is_open())
        {
            file << ros_image.header.stamp << " ";
            file << mean_state.transpose() << std::endl;
            file.close();
        }
        else
        {
            std::cout << "could not open file " << path_ << std::endl;
            exit(-1);
        }
    }

private:
    boost::shared_ptr<RobotTracker> tracker_;
    ros::NodeHandle node_handle_;
    boost::filesystem::path path_;
};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_filter");
    ros::NodeHandle node_handle("~");

    // read parameters
    std::cout << "reading parameters" << std::endl;
    std::string depth_image_topic; ri::ReadParameter("depth_image_topic", depth_image_topic, node_handle);
    std::string camera_info_topic; ri::ReadParameter("camera_info_topic", camera_info_topic, node_handle);
    int initial_sample_count; ri::ReadParameter("initial_sample_count", initial_sample_count, node_handle);
    double min_delta_time; ri::ReadParameter("min_delta_time", min_delta_time, node_handle);


    std::string source; ri::ReadParameter("source", source, node_handle);

    TrackingDataset TrackingDataset(source);
    std::cout << "laoding bagfile " << std::endl;
    TrackingDataset.Load();
    std::cout << "done" << std::endl;

    std::cout << "setting initial state " << std::endl;
    std::cout << TrackingDataset.GetGroundTruth(0).transpose() << std::endl;
    std::cout << "done printing vector " << std::endl;
    std::vector<Eigen::VectorXd> initial_states(1, TrackingDataset.GetGroundTruth(0));

    std::cout << "initializing filter " << std::endl;
    // intialize the filter

    boost::shared_ptr<RobotTracker> tracker(new RobotTracker);
    boost::shared_ptr<KinematicsFromURDF> urdf_kinematics(new KinematicsFromURDF());
    tracker->Initialize(initial_states, *TrackingDataset.GetImage(0), TrackingDataset.GetCameraMatrix(0), urdf_kinematics);
    TrackerInterface interface(tracker);

    ros::Publisher image_publisher = node_handle.advertise<sensor_msgs::Image>("/bagfile/depth/image", 0);
    ros::Publisher cloud_publisher = node_handle.advertise<pcl::PointCloud<pcl::PointXYZ> > ("/bagfile/depth/points", 0);

    std::cout << "processing TrackingDataset of Size: " << TrackingDataset.Size() << std::endl;
    for(size_t i = 0; i < TrackingDataset.Size() && ros::ok(); i++)
    {
        INIT_PROFILING
        double start_time; GET_TIME(start_time);

        interface.FilterAndStore(*TrackingDataset.GetImage(i));
        image_publisher.publish(*TrackingDataset.GetImage(i));
        cloud_publisher.publish((*TrackingDataset.GetPointCloud(i)).makeShared());

        double end_time; GET_TIME(end_time);
        while(end_time - start_time < min_delta_time)
            GET_TIME(end_time);
        MEASURE("========================================================>>>>>>>>> ");


        std::cout << "time for frame " << i << ": " << end_time - start_time << std::endl;
    }
    std::cout << std::endl << "done processing TrackingDataset" << std::endl;

    return 0;
}
