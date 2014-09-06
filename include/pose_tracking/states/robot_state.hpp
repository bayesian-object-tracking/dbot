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

#include <pose_tracking/states/rigid_bodies_state.hpp>

// TODO: THERE IS A PROBLEM HERE BECAUSE WE SHOULD NOT DEPEND ON THIS FILE,
// SINCE IT IS IN A PACKAGE WHICH IS BELOW THIS PACKAGE.
#include <pose_tracking_interface/utils/kinematics_from_urdf.hpp>

template<int size_joints>
struct RobotStateTypes
{
  enum
    {
      // For the joints
      JOINT_ANGLE_INDEX = 0,
      //JOINT_VELOCITY_INDEX = 1,
      // TODO: Check if obsolete
      // for the links
      COUNT_PER_BODY = 12,
      BLOCK_COUNT = 3,
      POSITION_INDEX = 0,
      ORIENTATION_INDEX = 3,
      LINEAR_VELOCITY_INDEX = 6,
      ANGULAR_VELOCITY_INDEX = 9
    };

  typedef ff::RigidBodiesState<size_joints> Base;
};


template<int size_joints = -1, int size_bodies = -1>
class RobotState: public RobotStateTypes<size_joints>::Base
{
public:
  typedef RobotStateTypes<size_joints> Types;
  typedef typename Types::Base Base;

  typedef typename Base::Scalar   Scalar;
  typedef typename Base::State    State;
  typedef typename Base::Vector   Vector;

  typedef typename Base::AngleAxis            AngleAxis;
  typedef typename Base::Quaternion           Quaternion;
  typedef typename Base::RotationMatrix       RotationMatrix;
  typedef typename Base::HomogeneousMatrix    HomogeneousMatrix;

  enum
    {
      // for the joints
      JOINT_ANGLE_INDEX = Types::JOINT_ANGLE_INDEX,
      //JOINT_VELOCITY_INDEX = Types::JOINT_VELOCITY_INDEX,
      // For the links
      SIZE_BODIES = size_bodies,
      COUNT_PER_BODY = Types::COUNT_PER_BODY,
      BLOCK_COUNT = Types::BLOCK_COUNT,
      POSITION_INDEX = Types::POSITION_INDEX,
      ORIENTATION_INDEX = Types::ORIENTATION_INDEX,
      LINEAR_VELOCITY_INDEX = Types::LINEAR_VELOCITY_INDEX,
      ANGULAR_VELOCITY_INDEX = Types::ANGULAR_VELOCITY_INDEX
    };

  // give access to base member functions (otherwise it is shadowed)
  //using Base::quaternion;
  //using Base::state_size;

  //TODO: SHOULD THIS BE ALLOWED?
  using Base::operator=;
  
  RobotState(): initialized_(false) {  }

  // constructor for dynamic size without initial value

  // TODO: COULD WE GET THE NUMBER OF BODIES AND THE NUMBER OF JOINTS FROM THE KINEMATICS?
  RobotState(unsigned body_count,
             const boost::shared_ptr<KinematicsFromURDF>& kinematics):
                                                Base(State::Zero(kinematics->num_joints())),
                                                body_count_(body_count),
                                                kinematics_(kinematics),
                                                initialized_(true)
  {
  }

  // constructor with initial value
  template <typename T> RobotState(const Eigen::MatrixBase<T>& state_vector):
    Base(state_vector)
  { }


  virtual ~RobotState() {}

  virtual void update() const
  {
    CheckInitialization("update");
    kinematics_->InitKDLData(*this);
  }


  virtual Vector position(const size_t& object_index = 0) const
  {
    CheckInitialization("position");
    return kinematics_->GetLinkPosition(object_index);
  }

  virtual Vector euler_vector(const size_t& object_index = 0) const
  {
    CheckInitialization("euler_vector");
    return Quaternion2EulerVector(kinematics_->GetLinkOrientation(object_index));
  }

  virtual Vector Quaternion2EulerVector(const Quaternion& quaternion) const 
  {
    AngleAxis angle_axis(quaternion);
    return angle_axis.angle()*angle_axis.axis();
  }


  virtual unsigned body_count() const
  {
      CheckInitialization("bodies_size");

      return kinematics_->num_links();
  }

//  virtual unsigned joints_size() const
//  {
//    CheckInitialization("joints_size");
//    return joint_count_;
//  }


  void GetJointState(std::map<std::string, double>& joint_positions)
  {
      CheckInitialization("GetJointState");
      std::vector<std::string> joint_map = kinematics_->GetJointMap();
      for(std::vector<std::string>::const_iterator it = joint_map.begin(); it != joint_map.end(); ++it)
      {
          joint_positions[*it] = (*this)(it - joint_map.begin(),0);
      }
  }


private:
  void CheckInitialization(const std::string &func) const
  {
      if(!initialized_)
      {
          std::cout << func << " the kinematics were not passed"
                    << " in the constructor of robot state " << std::endl;
          exit(-1);
      }
  }

  bool initialized_;

  unsigned body_count_;


  // pointer to the robot kinematic
  boost::shared_ptr<KinematicsFromURDF>  kinematics_;
};



#endif
