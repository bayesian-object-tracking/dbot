
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <ctime>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include <boost/filesystem.hpp>

#include <ff/utils/profiling.hpp>

#include <pose_tracking_interface/trackers/object_tracker.hpp>
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

    fl::FreeFloatingRigidBodiesState<-1> mean_state(1);

    // alternative initialization
    std::vector<Eigen::VectorXd> initial_states =
            pi::SampleTableClusters(fl::hf::Image2Points(image, camera_matrix),
                                    1000);
    boost::shared_ptr<MultiObjectTracker> tracker_particle_filter(new MultiObjectTracker);
    tracker_particle_filter->Initialize(initial_states, *ros_image, camera_matrix);
    sensor_msgs::Image fuck_you = *ros_image;
    for (int var = 0; var < 50  && ros::ok(); ++var)
    {
        fuck_you.header.stamp += ros::Duration(1./30.);
        mean_state = tracker_particle_filter->Filter(fuck_you);
    }

    // calib_obj
//    mean_state.setZero();
//    mean_state(0, 0) = 0.119531491;
//    mean_state(1, 0) = 0.040621002;
//    mean_state(2, 0) = 0.838543503;
//    mean_state(3, 0) = 0.560296146;
//    mean_state(4, 0) = 2.564343082;
//    mean_state(5, 0) = -1.352442605;

    std::cout << mean_state << std::endl;

//    // intialize the filter
    boost::shared_ptr<FukfTestTracker> tracker(new FukfTestTracker());

    std::cout << "tracker->Initialize" << std::endl;
    tracker->Initialize(mean_state,
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
