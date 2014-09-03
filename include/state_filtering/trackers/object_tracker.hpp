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

#ifndef STATE_FILTERING_TRACKERS_OBJECT_TRACKER_HPP
#define STATE_FILTERING_TRACKERS_OBJECT_TRACKER_HPP

//#define PROFILING_ON

#include <boost/thread/mutex.hpp>

#include <Eigen/Dense>

#include <vector>
#include <string>

// ros stuff
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <state_filtering/filters/stochastic/rao_blackwell_coordinate_particle_filter.hpp>
#include <state_filtering/models/observers/image_observer_cpu.hpp>
#ifdef BUILD_GPU
#include <state_filtering/models/observers/image_observer_gpu/image_observer_gpu.hpp>
#endif

class MultiObjectTracker
{
public:   
    typedef sf::FloatingBodySystem<> State;
    typedef State::Scalar        Scalar;

    typedef sf::BrownianObjectMotion<State>     ProcessModel;
    typedef sf::ImageObserverCPU<Scalar, State> ObserverCPUType;

#ifdef BUILD_GPU
    typedef sf::ImageObserverGPU<State>  ObserverGPUType;
#endif

    typedef ObserverCPUType::Base ObservationModel;
    typedef ObserverCPUType::Observation Observation;

    typedef sf::RaoBlackwellCoordinateParticleFilter<ProcessModel, ObservationModel> FilterType;

    MultiObjectTracker();

    void Initialize(std::vector<Eigen::VectorXd> initial_states,
                    const sensor_msgs::Image& ros_image,
                    Eigen::Matrix3d camera_matrix,
                    bool state_is_partial = true);

    Eigen::VectorXd Filter(const sensor_msgs::Image& ros_image);

private:  
    Scalar last_measurement_time_;

    boost::mutex mutex_;
    ros::NodeHandle node_handle_;
    ros::Publisher object_publisher_;

    boost::shared_ptr<FilterType> filter_;

    // parameters
    std::vector<std::string> object_names_;
    int downsampling_factor_;
};


#endif

