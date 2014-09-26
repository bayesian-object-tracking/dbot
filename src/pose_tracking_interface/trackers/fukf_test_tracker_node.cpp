

#include <iostream>
#include <fstream>
#include <ctime>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include <boost/filesystem.hpp>

#include <fast_filtering/utils/profiling.hpp>

#include <pose_tracking_interface/trackers/fukf_test_tracker.hpp>
#include <pose_tracking_interface/utils/pcl_interface.hpp>
#include <pose_tracking_interface/utils/ros_interface.hpp>

typedef sensor_msgs::CameraInfo::ConstPtr CameraInfoPtr;
typedef Eigen::Matrix<double, -1, -1> Image;

//class Tracker
//{
//public:
//    Tracker(boost::shared_ptr<FukfTestTracker> tracker): tracker_(tracker), node_handle_("~")
//    {
////        std::string config_file;
////        ri::ReadParameter("config_file", config_file, node_handle_);

////        path_ = config_file;
////        path_ = path_.parent_path();
////        std::cout << path_ << std::endl;

////        time_t rawtime;
////        struct tm * timeinfo;
////        char buffer[80];

////        time (&rawtime);
////        timeinfo = localtime(&rawtime);

////        strftime(buffer,80,"%d.%m.%Y_%I.%M.%S",timeinfo);
////        std::string current_time(buffer);

////        path_ /= "tracking_data_" + current_time + ".txt";
//    }
//    ~Tracker() {}

//    void Filter(const sensor_msgs::Image& ros_image)
//    {
//        INIT_PROFILING
//        ff::FreeFloatingRigidBodiesState<-1> mean_state = tracker_->Filter(ros_image);
//        MEASURE("total time for filtering")
//    }

//    void FilterAndStore(const sensor_msgs::Image& ros_image)
//    {
//        INIT_PROFILING
//        ff::FreeFloatingRigidBodiesState<-1> mean_state = tracker_->Filter(ros_image);
//        MEASURE("total time for filtering")

//        std::ofstream file;
//        file.open(path_.c_str(), std::ios::out | std::ios::app);
//        if(file.is_open())
//        {
//            file << ros_image.header.stamp << " ";
//            file << mean_state.poses().transpose() << std::endl;
//            file.close();
//        }
//        else
//        {
//            std::cout << "could not open file " << path_ << std::endl;
//            exit(-1);
//        }
//    }

//private:
//    boost::shared_ptr<FukfTestTracker> tracker_;
//    ros::NodeHandle node_handle_;
//    boost::filesystem::path path_;
//};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "fukf_test_filter");
    ros::NodeHandle node_handle("~");

    // read parameters
    std::string depth_image_topic;
    std::string camera_info_topic;
    std::string source;
    std::vector<std::string> object_names;

    std::cout << "reading parameters" << std::endl;
    ri::ReadParameter("camera_info_topic", camera_info_topic, node_handle);
    ri::ReadParameter("depth_image_topic", depth_image_topic, node_handle);
    ri::ReadParameter("object_names", object_names, node_handle);
    ri::ReadParameter("source", source, node_handle);


//    // read from camera
//    if(source == "camera")
//    {
//        std::cout << "reading data from camera " << std::endl;
//        Eigen::Matrix3d camera_matrix = ri::GetCameraMatrix<double>(camera_info_topic, node_handle, 2.0);

//        // get observations from camera
//        sensor_msgs::Image::ConstPtr ros_image =
//                ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic, node_handle, ros::Duration(10.0));
//        Image image = ri::Ros2Eigen<double>(*ros_image);

//        std::vector<Eigen::VectorXd> initial_states = pi::SampleTableClusters(ff::hf::Image2Points(image, camera_matrix),
//                                                                  initial_sample_count);

//        // intialize the filter
//        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
//        tracker->Initialize(initial_states, *ros_image, camera_matrix);
//        std::cout << "done initializing" << std::endl;
//        Tracker interface(tracker);

//        ros::Subscriber subscriber = node_handle.subscribe(depth_image_topic, 1, &Tracker::FilterAndStore, &interface);
//        ros::spin();
//    }
//    // read from bagfile
//    else
//    {
//        TrackingDataset TrackingDataset(source);
//        std::cout << "laoding bagfile " << std::endl;
//        TrackingDataset.Load();
//        std::cout << "done" << std::endl;

//        std::cout << "setting initial state " << std::endl;
//        std::cout << TrackingDataset.GetGroundTruth(0).transpose() << std::endl;
//        std::cout << "done printing vector " << std::endl;
//        ff::FreeFloatingRigidBodiesState<-1> initial_state(object_names.size());
//        initial_state.poses(TrackingDataset.GetGroundTruth(0).topRows(object_names.size()*6)); // we read only the part of the state we need
//        std::vector<Eigen::VectorXd> initial_states(1, initial_state);

//        std::cout << "initializing filter " << std::endl;
//        // intialize the filter
//        boost::shared_ptr<MultiObjectTracker> tracker(new MultiObjectTracker);
//        tracker->Initialize(initial_states, *TrackingDataset.GetImage(0), TrackingDataset.GetCameraMatrix(0), false);
//        Tracker interface(tracker);

//        ros::Publisher image_publisher = node_handle.advertise<sensor_msgs::Image>("/bagfile/depth/image", 0);
//        ros::Publisher cloud_publisher = node_handle.advertise<pcl::PointCloud<pcl::PointXYZ> > ("/bagfile/depth/points", 0);

//        std::cout << "processing TrackingDataset of Size: " << TrackingDataset.Size() << std::endl;
//        for(size_t i = 0; i < TrackingDataset.Size() && ros::ok(); i++)
//        {
//            interface.FilterAndStore(*TrackingDataset.GetImage(i));
//            image_publisher.publish(*TrackingDataset.GetImage(i));
//            cloud_publisher.publish((*TrackingDataset.GetPointCloud(i)).makeShared());
//        }
//        std::cout << std::endl << "done processing TrackingDataset" << std::endl;
//    }

    return 0;
}
