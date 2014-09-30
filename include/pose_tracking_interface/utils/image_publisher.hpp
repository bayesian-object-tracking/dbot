
#ifndef TOOLS_DEPTH_IMAGE_PUBLISHER_
#define TOOLS_DEPTH_IMAGE_PUBLISHER_

#include <ros/ros.h>
#include <string>
#include <map>
#include <inttypes.h>
#include <Eigen/Dense>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv/cvwimage.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ff
{
class ImagePublisher
{
public:
    ImagePublisher(ros::NodeHandle& node_handle);

    void publish(const Eigen::MatrixXd& m,
                 const std::string& name,
                 int height,
                 int width,
                 float min,
                 float max,
                 bool autoCreatePublisher = true);

    void addPublisher(const std::string name);

    bool hasPublisher(const std::string& name);

    sensor_msgs::ImagePtr measurementToRosImage(const Eigen::MatrixXd& m,
                                                int height,
                                                int width,
                                                float min,
                                                float max);
    /**
     * @brief rainbow_init Taken from arm_rgbd package
     */
    void rainbow_init();

    /**
     * @brief rainbow_set Taken from arm_rgbd package and slightly adapted to OpenCV
     */
    void rainbow_set(cv::Vec3b& dst, float value, float min, float max);

    sensor_msgs::ImagePtr measurementToRosImagePT(const Eigen::MatrixXd &m, int height, int width, float min, float max);
    void publish(const Eigen::MatrixXd &m, const std::string &name, int height, int width, bool autoCreatePublisher = true);
    sensor_msgs::ImagePtr measurementToRosImage(const Eigen::MatrixXd &m, int height, int width);
protected:
        uint8_t rainbow[0x10000][3];
        std::map<std::string, image_transport::Publisher> publisher;
        std::map<std::string, sensor_msgs::ImagePtr> images;
        image_transport::ImageTransport it;
};
}

#endif
