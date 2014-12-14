


#include <pose_tracking_interface/utils/image_publisher.hpp>

using namespace fl;

ImagePublisher::ImagePublisher(ros::NodeHandle& node_handle):
    it(image_transport::ImageTransport(node_handle))
{
    rainbow_init();
}

void ImagePublisher::publish(const Eigen::MatrixXd& m,
             const std::string& name,
             int height,
             int width,
             float min,
             float max,
             bool autoCreatePublisher)
{
    if (!hasPublisher(name) && !autoCreatePublisher)
    {
        ROS_DEBUG_STREAM("Image publisher '" << name << "' doesn't exist." );

        return;
    }
    else if (!hasPublisher(name) && autoCreatePublisher)
    {
        addPublisher(name);
    }

    //std::cout << "publishing " << name << std::endl;
    images[name] = toRosImage(m, height, width, min, max);
    publisher[name].publish(images[name]);
}

void ImagePublisher::publish(const Eigen::MatrixXd& m,
             const std::string& name,
             int height,
             int width,
             bool autoCreatePublisher)
{
    if (!hasPublisher(name) && !autoCreatePublisher)
    {
        ROS_DEBUG_STREAM("Image publisher '" << name << "' doesn't exist." );

        return;
    }
    else if (!hasPublisher(name) && autoCreatePublisher)
    {
        addPublisher(name);
    }

    //std::cout << "publishing " << name << std::endl;
    images[name] = toRosImagePT(m, height, width);
    publisher[name].publish(images[name]);
}

void ImagePublisher::addPublisher(const std::string name)
{
    if (hasPublisher(name))
    {
        ROS_DEBUG_STREAM("Image publisher '" << name << "' already exist." );

        return;
    }

    publisher[name] = it.advertise(name, 1);
}

bool ImagePublisher::hasPublisher(const std::string& name)
{
    return (publisher.find(name) != publisher.end());
}

sensor_msgs::ImagePtr ImagePublisher::toRosImage(const Eigen::MatrixXd& m,
                                            int height,
                                            int width,
                                            float min,
                                            float max)
{    
    cv_bridge::CvImage cvImage;

    cvImage.image = cv::Mat(height, width, CV_8UC3);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            rainbow_set(cvImage.image.at<cv::Vec3b>(i, j), m(i * width + j, 0), min, max);
        }
    }

    cvImage.encoding = "rgb8";
    return cvImage.toImageMsg();
}

sensor_msgs::ImagePtr ImagePublisher::toRosImage(const Eigen::MatrixXd& m,
                                            int height,
                                            int width)
{
    cv_bridge::CvImage cvImage;

    float min_val = 65535;
    float max_val = 0;
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            float val = m(i * width + j, 0);

            if(val < min_val)
            {
              min_val = val;
            }
            if(val > max_val)
            {
              max_val = val;
            }
        }
    }

    float factor = (min_val < 1.0f && min_val> 0.0f) ?  1.0f/min_val: 1;
    //float shift = (min_val < 0 ? -min_val: 0);
    float shift = -min_val;

    cvImage.image = cv::Mat(height, width, CV_8UC3);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            rainbow_set(cvImage.image.at<cv::Vec3b>(i, j),
                        factor*m(i * width + j, 0) + shift,
                        min_val*factor + shift,
                        max_val*factor + shift);
        }
    }

    /*
    if (factor > 1.0f)
    {
        std::cout << "factoooor " << factor << " :: " << min_val*factor + shift << "/" << max_val*factor + shift << std::endl;
    }

    if (shift > 0.0f)
    {
        std::cout << "shiffffft " << shift << " :: " << min_val*factor + shift << "/" << max_val*factor + shift << std::endl;
    }
    */

    cvImage.encoding = "rgb8";
    return cvImage.toImageMsg();
}


sensor_msgs::ImagePtr ImagePublisher::toRosImagePT(const Eigen::MatrixXd& m,
                                            int height,
                                            int width)
{
    cv_bridge::CvImage cvImage;

    float min_val =  1.e10f;
    float max_val = -1.e10f;
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            float val = m(i * width + j, 0);

            if(val < min_val)
            {
              min_val = val;
            }
            if(val > max_val)
            {
              max_val = val;
            }
        }
    }

    cvImage.image = cv::Mat(height, width, CV_8UC1);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            cvImage.image.at<unsigned char>(i, j) = (m(i * width + j, 0) - min_val) / (max_val-min_val) * 255;
        }
    }

    //cvImage.image.convertTo(cvImage.image, CV_16SC1); //, (65535.0)/(max_val - min_val), - min_val*(65535.0)/(max_val - min_val));
    cvImage.encoding = "mono8";

    return cvImage.toImageMsg();
}


void ImagePublisher::rainbow_init()
{
    int i;
    float d;

    // This is the non-linearized rainbow scale.
    for (i = 0; i < 0x10000; i++)
    {
        d = 4 * ((double) i / 0x10000);

        if (d >= 0 && d < 1.0)
        {
            rainbow[i][0] = 0x00;
            rainbow[i][1] = (int) (d * 0xFF);
            rainbow[i][2] = 0xFF;
        }
        else if (d < 2.0)
        {
            d -= 1.0;
            rainbow[i][0] = 0x00;
            rainbow[i][1] = 0xFF;
            rainbow[i][2] = (int) ((1 - d) * 0xFF);
        }
        else if (d < 3.0)
        {
            d -= 2.0;
            rainbow[i][0] = (int) (d * 0xFF);
            rainbow[i][1] = 0xFF;
            rainbow[i][2] = 0x00;
        }
        else if (d < 4.0)
        {
            d -= 3.0;
            rainbow[i][0] = 0xFF;
            rainbow[i][1] = (int) ((1 - d) * 0xFF);
            rainbow[i][2] = 0x00;
        }
        else
        {
            rainbow[i][0] = 0xFF;
            rainbow[i][1] = 0x00;
            rainbow[i][2] = 0x00;
        }
    }

    return;
}

// Set a pixel using then rainbow scale.
void ImagePublisher::rainbow_set(cv::Vec3b& dst, float value, float min, float max)
{
    int k;

    if (value < min)
        value = min;
    if (value > max)
        value = max;    

    k = (int) ((value - min) / (max - min) * 0xFFFF);

    if (k < 0 || k > 0xFFFF)
    {
        dst[0] = 0x80;
        dst[1] = 0x80;
        dst[2] = 0x80;
        return;
    }

    dst[0] = rainbow[k][0];
    dst[1] = rainbow[k][1];
    dst[2] = rainbow[k][2];

    return;
}
