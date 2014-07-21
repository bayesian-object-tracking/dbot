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

#include <sensor_msgs/Image.h>

#include <state_filtering/multi_object_tracker.hpp>
#include <state_filtering/tools/cloud_visualizer.hpp>


#include <cv.h>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <boost/filesystem.hpp>

#include <ctime>


#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include "state_filtering/system_states/rigid_body_system.hpp"

#include <state_filtering/tools/tracking_dataset.hpp>

#include <pcl_ros/point_cloud.h>




typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;
typedef Eigen::Matrix<double, -1, -1> Image;













class TrackerInterface
{
public:
    TrackerInterface(boost::shared_ptr<MultiObjectTracker> tracker): tracker_(tracker), node_handle_("~")
    {
        string config_file;
        ri::ReadParameter("config_file", config_file, node_handle_);

        path_ = config_file;
        path_ = path_.parent_path();
        cout << path_ << endl;

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
        FloatingBodySystem<-1> mean_state = tracker_->Filter(ros_image);
        MEASURE("total time for filtering")
    }

    void FilterAndStore(const sensor_msgs::Image& ros_image)
    {
        INIT_PROFILING
        FloatingBodySystem<-1> mean_state = tracker_->Filter(ros_image);
        MEASURE("total time for filtering")

        ofstream file;
        file.open(path_.c_str(), ios::out | ios::app);
        if(file.is_open())
        {
            file << ros_image.header.stamp << " ";
            file << mean_state.poses().transpose() << endl;
            file.close();
        }
        else
        {
            cout << "could not open file " << path_ << endl;
            exit(-1);
        }
    }




private:
    boost::shared_ptr<MultiObjectTracker> tracker_;
    ros::NodeHandle node_handle_;
    boost::filesystem::path path_;
};



int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_filter");
    ros::NodeHandle node_handle("~");

    // read parameters
    cout << "reading parameters" << endl;
    string depth_image_topic; ri::ReadParameter("depth_image_topic", depth_image_topic, node_handle);
    string camera_info_topic; ri::ReadParameter("camera_info_topic", camera_info_topic, node_handle);
    string source; ri::ReadParameter("source", source, node_handle);
    vector<string> object_names; ri::ReadParameter("object_names", object_names, node_handle);

    int initial_sample_count; ri::ReadParameter("initial_sample_count", initial_sample_count, node_handle);

    // read from camera
    if(source == "camera")
    {
        cout << "reading data from camera " << endl;
        Matrix3d camera_matrix = ri::GetCameraMatrix<double>(camera_info_topic, node_handle, 2.0);

        // get observations from camera
        sensor_msgs::Image::ConstPtr ros_image =
                ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic, node_handle, ros::Duration(10.0));
        Image image = ri::Ros2Eigen<double>(*ros_image);

        vector<VectorXd> initial_states = pi::SampleTableClusters(hf::Image2Points(image, camera_matrix),
                                                                  initial_sample_count);

        // intialize the filter
        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
        tracker->Initialize(initial_states, *ros_image, camera_matrix);
        cout << "done initializing" << endl;
        TrackerInterface interface(tracker);

        ros::Subscriber subscriber = node_handle.subscribe(depth_image_topic, 1, &TrackerInterface::FilterAndStore, &interface);
        ros::spin();
    }
    // read from bagfile
    else
    {
        TrackingDataset TrackingDataset(source);
        cout << "laoding bagfile " << endl;
        TrackingDataset.loAd();
        cout << "done" << endl;

        cout << "setting initial state " << endl;
        cout << TrackingDataset.getGroundTruth(0).transpose() << endl;
        cout << "done printing vector " << endl;
        FloatingBodySystem<-1> initial_state(object_names.size());
        initial_state.poses(TrackingDataset.getGroundTruth(0).topRows(object_names.size()*6)); // we read only the part of the state we need
        vector<VectorXd> initial_states(1, initial_state);

        cout << "initializing filter " << endl;
        // intialize the filter
        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
        tracker->Initialize(initial_states, *TrackingDataset.getImage(0), TrackingDataset.getCameraMatrix(0), false);
        TrackerInterface interface(tracker);

        ros::Publisher image_publisher = node_handle.advertise<sensor_msgs::Image>("/bagfile/depth/image", 0);
        ros::Publisher cloud_publisher = node_handle.advertise<pcl::PointCloud<pcl::PointXYZ> > ("/bagfile/depth/points", 0);

        cout << "processing TrackingDataset of size: " << TrackingDataset.sIze() << endl;
        for(size_t i = 0; i < TrackingDataset.sIze() && ros::ok(); i++)
        {
            interface.FilterAndStore(*TrackingDataset.getImage(i));
            image_publisher.publish(*TrackingDataset.getImage(i));
            cloud_publisher.publish((*TrackingDataset.getPointCloud(i)).makeShared());
        }
        cout << endl << "done processing TrackingDataset" << endl;
    }



    return 0;
}
