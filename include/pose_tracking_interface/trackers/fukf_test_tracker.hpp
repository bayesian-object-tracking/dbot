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

#ifndef POSE_TRACKING_INTERFACE_FUKF_TEST_TRACKER_HPP
#define POSE_TRACKING_INTERFACE_FUKF_TEST_TRACKER_HPP

#include <boost/thread/mutex.hpp>

#include <Eigen/Dense>

#include <vector>
#include <string>

// ros stuff
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <fast_filtering/filters/deterministic/composed_state_distribution.hpp>
#include <fast_filtering/filters/deterministic/factorized_unscented_kalman_filter.hpp>
#include <pose_tracking/models/process_models/continuous_occlusion_process_model.hpp>
#include <pose_tracking/models/process_models/brownian_object_motion_model.hpp>
#include <pose_tracking/models/observation_models/continuous_kinect_pixel_observation_model.hpp>
#include <pose_tracking/models/observation_models/approximate_kinect_pixel_observation_model.hpp>

#include <pose_tracking_interface/utils/image_publisher.hpp>
#include <pose_tracking_interface/utils/image_visualizer.hpp>

class FukfTestTracker
{
public:
    typedef ff::FreeFloatingRigidBodiesState<> State_a;
    typedef State_a::Scalar                    Scalar;

    typedef ff::BrownianObjectMotionModel<State_a> ProcessModel_a;
    typedef ff::ContinuousOcclusionProcessModel    ProcessModel_b;

    typedef ff::ApproximateKinectPixelObservationModel<State_a> ObservationModel;
//    typedef ff::ContinuousKinectPixelObservationModel<State_a> ObservationModel;

    typedef ProcessModel_b::State State_b;
    typedef ObservationModel::Observation Observation;

    typedef ff::ComposedStateDistribution<State_a, State_b> StateDistribution;

    typedef ff::FactorizedUnscentedKalmanFilter<ProcessModel_a,
                                                ProcessModel_b,
                                                ObservationModel> FilterType;

    FukfTestTracker();

    void Initialize(State_a initial_state,
                    const sensor_msgs::Image& ros_image,
                    Eigen::Matrix3d camera_matrix);

    void Filter(const sensor_msgs::Image& ros_image);

private:  
    Scalar last_measurement_time_;

    boost::mutex mutex_;
    ros::NodeHandle nh_;
    ros::Publisher object_publisher_;

    boost::shared_ptr<FilterType> filter_;

    // parameters
    std::vector<std::string> object_names_;
    int downsampling_factor_;

    StateDistribution state_distr;

    ff::ImagePublisher ip_;
    int rows_;
    int cols_;
};


#endif

