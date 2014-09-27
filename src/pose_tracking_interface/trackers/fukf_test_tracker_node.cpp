
#include <Eigen/Dense>

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

int main (int argc, char **argv)
{
    ros::init(argc, argv, "fukf_test_filter");
    ros::NodeHandle node_handle("~");

    // read parameters
    std::string depth_image_topic;
    std::string camera_info_topic;
    int initial_sample_count;

    std::cout << "reading parameters" << std::endl;
    ri::ReadParameter("camera_info_topic", camera_info_topic, node_handle);
    ri::ReadParameter("depth_image_topic", depth_image_topic, node_handle);
    ri::ReadParameter("initial_sample_count", initial_sample_count, node_handle);


    std::cout << "reading data from camera " << std::endl;
    Eigen::Matrix3d camera_matrix =
            ri::GetCameraMatrix<double>(camera_info_topic, node_handle, 2.0);

    // get observations from camera
    sensor_msgs::Image::ConstPtr ros_image =
            ros::topic::waitForMessage<sensor_msgs::Image>(depth_image_topic,
                                                           node_handle,
                                                           ros::Duration(10.0));
    Image image = ri::Ros2Eigen<double>(*ros_image);

    std::cout << "ri::Ros2Eigen" << std::endl;

    std::vector<Eigen::VectorXd> initial_states =
            pi::SampleTableClusters(ff::hf::Image2Points(image, camera_matrix),
                                    initial_sample_count);

    std::cout << "new tracker" << std::endl;

    // intialize the filter
    boost::shared_ptr<FukfTestTracker> tracker(new FukfTestTracker());

    std::cout << "tracker created" << std::endl;

//    // FIXME replace this with the result of the particle filter
    std::cout << "mean of " << initial_states.size() << " initial states " << std::endl;
    Eigen::VectorXd mean_initial_state;
    mean_initial_state = Eigen::VectorXd::Zero(initial_states[0].rows(), 1);
    for (auto& initial_state: initial_states)
    {
        mean_initial_state += initial_state;
    }
    mean_initial_state /= double(initial_states.size());

    std::cout << "tracker->Initialize" << std::endl;
    tracker->Initialize(mean_initial_state,
                        *ros_image,
                        camera_matrix);

    std::cout << "subscribing to depth_image_topic " << std::endl;
    ros::Subscriber subscriber = node_handle.subscribe(
                depth_image_topic,
                1,
                &FukfTestTracker::Filter,
                tracker);
    ros::spin();

    return 0;
}
