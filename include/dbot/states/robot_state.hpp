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


#ifndef POSE_TRACKING_STATES_ROBOT_STATE_HPP
#define POSE_TRACKING_STATES_ROBOT_STATE_HPP

#include <Eigen/Dense>
#include <vector>

#include <dbot/states/rigid_bodies_state.hpp>

// TODO: THERE IS A PROBLEM HERE BECAUSE WE SHOULD NOT DEPEND ON THIS FILE,
// SINCE IT IS IN A PACKAGE WHICH IS BELOW THIS PACKAGE.
#include <dbot_ros_pkg/utils/kinematics_from_urdf.hpp>


template<int JointCount = Eigen::Dynamic, int BodyCount = Eigen::Dynamic>
class RobotState: public fl::RigidBodiesState<JointCount>
{
public:
    typedef fl::RigidBodiesState<JointCount>    Base;
    typedef typename Base::Vector               Vector;
    typedef typename Base::AngleAxis            AngleAxis;
    typedef typename Base::Quaternion           Quaternion;

public:
    RobotState(): Base() { }

    template <typename T>
    RobotState(const Eigen::MatrixBase<T>& state_vector): Base(state_vector) { }

    virtual ~RobotState() { }

    using Base::operator=;

    virtual Vector position(const size_t& object_index = 0) const
    {
        CheckKinematics();
        kinematics_->InitKDLData(*this);
        return kinematics_->GetLinkPosition(object_index);
    }

    virtual Vector euler_vector(const size_t& object_index = 0) const
    {
        CheckKinematics();
        kinematics_->InitKDLData(*this);
        return Quaternion2EulerVector(kinematics_->GetLinkOrientation(object_index));
    }

    virtual Quaternion quaternion(const size_t& object_index = 0) const
    {
        return kinematics_->GetLinkOrientation(object_index);
    }

    virtual unsigned body_count() const
    {
        CheckKinematics();
        return kinematics_->num_links();
    }

    // TODO: SHOULD THIS FUNCITON BE IN HERE?
    void GetJointState(std::map<std::string, double>& joint_positions)
    {
        CheckKinematics();
        std::vector<std::string> joint_map = kinematics_->GetJointMap();
        for(std::vector<std::string>::const_iterator it = joint_map.begin(); it != joint_map.end(); ++it)
        {
            joint_positions[*it] = (*this)(it - joint_map.begin(),0);
        }
    }

private:
    virtual Vector Quaternion2EulerVector(const Quaternion& quaternion) const
    {
        AngleAxis angle_axis(quaternion);
        return angle_axis.angle()*angle_axis.axis();
    }

    void CheckKinematics() const
    {
        if(!kinematics_)
        {
            std::cout << "kinematics not set" << std::endl;
            exit(-1);
        }
    }

public:
    static boost::shared_ptr<KinematicsFromURDF>  kinematics_;
};

template<int JointCount, int BodyCount>
boost::shared_ptr<KinematicsFromURDF> RobotState<JointCount, BodyCount>::kinematics_;


#endif
