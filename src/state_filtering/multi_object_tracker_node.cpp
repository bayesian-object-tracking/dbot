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

#include <boost/foreach.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include "state_filtering/system_states/rigid_body_system.hpp"



typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;
typedef Eigen::Matrix<double, -1, -1> Image;







class RangeData
{
public:
    sensor_msgs::Image::ConstPtr image_;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;

    RangeData(const sensor_msgs::Image::ConstPtr &image,
              const sensor_msgs::CameraInfo::ConstPtr &camera_info) :
        image_(image),
        camera_info_(camera_info)
    {}
};

/**
 * Inherits from message_filters::SimpleFilter<M>
 * to use protected signalMessage function
 */
template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const> &msg)
    {
        signalMessage(msg);
    }
};

// Callback for synchronized messages
class BagReader
{
public:
    BagReader() {}
    ~BagReader() {}

    void callback(const sensor_msgs::Image::ConstPtr &image,
                  const sensor_msgs::CameraInfo::ConstPtr &camera_info)
    {
        RangeData data(image, camera_info);
        data_.push_back(data);
    }

    // Load bag
    void loadBag(const std::string &filename, const std::string& image_topic, const std::string& camera_info_topic)
    {
        rosbag::Bag bag;
        bag.open(filename, rosbag::bagmode::Read);

        // Image topics to load
        std::vector<std::string> topics;
        topics.push_back(image_topic);
        topics.push_back(camera_info_topic);

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // Set up fake subscribers to capture images
        BagSubscriber<sensor_msgs::Image> image_subscriber;
        BagSubscriber<sensor_msgs::CameraInfo> camera_info_subscriber;

        // Use time synchronizer to make sure we get properly synchronized images
        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo>
                sync(image_subscriber, camera_info_subscriber, 25);
        sync.registerCallback(boost::bind(&BagReader::callback, this,  _1, _2));

        // Load all messages into our stereo dataset
        BOOST_FOREACH(rosbag::MessageInstance const m, view)
        {
            if (m.getTopic() == image_topic || ("/" + m.getTopic() == image_topic))
            {
                sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
                if (image != NULL)
                    image_subscriber.newMessage(image);
            }

            if (m.getTopic() == camera_info_topic || ("/" + m.getTopic() == camera_info_topic))
            {
                sensor_msgs::CameraInfo::ConstPtr camera_info = m.instantiate<sensor_msgs::CameraInfo>();
                if (camera_info != NULL)
                    camera_info_subscriber.newMessage(camera_info);
            }
        }
        bag.close();
    }

    vector<RangeData> data_;
};
















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
        cout << "filtering" << endl;
        double start_time; GET_TIME(start_time)
        FloatingBodySystem<-1> mean_state = tracker_->Filter(ros_image);
        double end_time; GET_TIME(end_time);
        double delta_time = end_time - start_time;
        cout << "delta time: " << delta_time << endl;
    }

    void FilterAndStore(const sensor_msgs::Image& ros_image)
    {
        double start_time; GET_TIME(start_time)
        FloatingBodySystem<-1> mean_state = tracker_->Filter(ros_image);
        double end_time; GET_TIME(end_time);
        double delta_time = end_time - start_time;
        cout << "delta time: " << delta_time << endl;

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
    int initial_sample_count; ri::ReadParameter("initial_sample_count", initial_sample_count, node_handle);

    // read from camera
    if(source == "camera")
    {
        cout << "reading data from camera " << endl;
        Matrix3d camera_matrix = ri::GetCameraMatrix<double>(camera_info_topic, node_handle, 2.0);

        // get observations from camera
        sensor_msgs::Image::ConstPtr ros_image =
                ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic, node_handle, ros::Duration(10.0));
        Image image = ri::Ros2Eigen<double>(*ros_image) / 1000.; // convert to m

        vector<VectorXd> initial_states = pi::SampleTableClusters(hf::Image2Points(image, camera_matrix),
                                                                  initial_sample_count);

        // intialize the filter
        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
        tracker->Initialize(initial_states, *ros_image, camera_matrix);
        cout << "done initializing" << endl;
        TrackerInterface interface(tracker);

        ros::Subscriber subscriber = node_handle.subscribe(depth_image_topic, 1, &TrackerInterface::Filter, &interface);
        ros::spin();
    }
    // read from bagfile
    else
    {
        // somehow the bagfile reader does not like the slash in the beginning
        if(depth_image_topic[0] == '/')
            depth_image_topic.erase(depth_image_topic.begin());
        cout << depth_image_topic << endl;
        if(camera_info_topic[0] == '/')
            camera_info_topic.erase(camera_info_topic.begin());

        cout << "reading data from bagfile " << endl;
        BagReader reader;
        reader.loadBag(source, depth_image_topic, camera_info_topic);
        cout << "data size: " << reader.data_.size() << endl;

        Matrix3d camera_matrix;
        for(size_t col = 0; col < 3; col++)
            for(size_t row = 0; row < 3; row++)
                camera_matrix(row,col) = reader.data_[0].camera_info_->K[col+row*3];

        // read initial state from file
        VectorXd poses;
        poses.resize(0,1);
        string config_file; ri::ReadParameter("config_file", config_file, node_handle);
        boost::filesystem::path path;
        path = config_file;
        path = path.parent_path().parent_path();
        path /= "ground_truth.txt";
        ifstream file;
        file.open(path.c_str(), ios::in);
        if(file.is_open())
        {
            std::string temp;
            std::getline(file, temp);
            std::istringstream line(temp);
            double scalar;
            line >> scalar; // first one is timestamp, we throw it away
            while(line >> scalar)
            {
                VectorXd temp(poses.rows() + 1);
                temp.topRows(poses.rows()) = poses;
                temp(temp.rows()-1) = scalar;
                poses = temp;
            }
            file.close();

            cout << "read initial state with size " << poses.rows() << endl;
        }
        else
        {
            cout << "could not open file " << path << endl;
            exit(-1);
        }

        FloatingBodySystem<-1> initial_state(poses.rows()/6);
        initial_state.poses(poses);
        vector<VectorXd> initial_states;
        initial_states.push_back(initial_state);

        // intialize the filter
        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
        tracker->Initialize(initial_states, *reader.data_[0].image_, camera_matrix, false);
        cout << "done initializing" << endl;
        TrackerInterface interface(tracker);

        ros::Publisher publisher = node_handle.advertise<sensor_msgs::Image>("/bagfile/depth/image", 0);

        for(size_t i = 0; i < reader.data_.size(); i++)
        {
            interface.FilterAndStore(*reader.data_[i].image_);
            publisher.publish(*reader.data_[i].image_);
        }
    }



    return 0;
}
